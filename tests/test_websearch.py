import json

import httpx
import pymupdf
import pytest

from simcheck import websearch
from simcheck.engine import CheckOptions, check_document
from simcheck.pdfreport import render_pdf
from simcheck.report import render_html
from simcheck.store import Store
from simcheck.textnorm import tokenize
from simcheck.websearch import (
    InvalidKey, RateLimited, SearchError, SearchResult, brave_search, make_query, matched_weight, parse_brave, web_pass,
)
from test_simcheck import SOURCE_EN, add_doc, no_integrity_data  # noqa: F401  （自動で使う設定も読み込む）

WEB_SENTENCE = (
    "The aqueous outflow of the trabecular meshwork was measured in twelve donor eyes with a perfusion system "
    "built in our laboratory"
)
OTHER_SENTENCE = (
    "Patients were followed for three years and the visual field was tested every six months by the same examiner "
    "in a quiet room"
)


@pytest.fixture(autouse=True)
def fast_and_isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(websearch, "INTERVAL_SECONDS", 0)
    monkeypatch.setattr(websearch, "RATE_LIMIT_WAIT", 0)
    monkeypatch.setattr(websearch, "CACHE_PATH", tmp_path / "web_cache.sqlite")


def result_with(text, url="https://example.org/page"):
    return SearchResult("Some page", url, f"Intro words. {text} More words.")


def fake_search(answers, calls=None):
    def search(query):
        if calls is not None:
            calls.append(query)
        for needle, results in answers.items():
            if needle in query:
                if isinstance(results, Exception):
                    raise results
                return results
        return []
    return search


def spans(text, *sentences):
    return [(text.index(s), text.index(s) + len(s)) for s in sentences]


def test_parse_brave_joins_extra_snippets():
    data = {"web": {"results": [
        {"title": "T", "url": "https://a.example", "description": "first", "extra_snippets": ["second", "third"]},
        {"title": "U", "url": "https://b.example"},
    ]}}
    results = parse_brave(data)
    assert results[0] == SearchResult("T", "https://a.example", "first … second … third")
    assert results[1].snippet == ""
    assert parse_brave({}) == []


def test_brave_search_sends_exact_phrase_and_maps_errors():
    seen = []

    def handler(request):
        seen.append(request)
        status = {"bad": 401, "busy": 429, "down": 500}.get(request.url.params["q"].strip('"'), 200)
        return httpx.Response(status, json={"web": {"results": [{"title": "T", "url": "https://x.example"}]}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert brave_search(client, "k", "some words")[0].url == "https://x.example"
    assert seen[0].url.params["q"] == '"some words"'
    assert seen[0].headers["X-Subscription-Token"] == "k"
    with pytest.raises(InvalidKey):
        brave_search(client, "k", "bad")
    with pytest.raises(RateLimited):
        brave_search(client, "k", "busy")
    with pytest.raises(SearchError):
        brave_search(client, "k", "down")


def test_make_query_drops_quotes_and_limits_length():
    query = make_query('He said "this is a test" and\nthen ' + " ".join(f"w{i}" for i in range(60)))
    assert '"' not in query and "\n" not in query
    assert len(tokenize(query)) == 30
    ja = make_query("「加齢黄斑変性」は" + "視力" * 100)
    assert "「" not in ja and sum(tokenize(ja).weights) <= websearch.MAX_QUERY_WEIGHT


def test_matched_weight_needs_the_same_word_sequence():
    query = "the aqueous outflow of the trabecular meshwork was measured"
    assert matched_weight(query, result_with(query)) == 9 * 5
    scrambled = SearchResult("t", "u", "meshwork trabecular the of outflow aqueous the measured was")
    assert matched_weight(query, scrambled) < 2 * 5


def test_web_pass_reports_hits_and_skips_common_and_unverified():
    text = f"{WEB_SENTENCE}. {OTHER_SENTENCE}. {SOURCE_EN}"
    answers = {
        make_query(WEB_SENTENCE): [result_with(WEB_SENTENCE), SearchResult("unrelated", "https://u.example", "nothing")],
        make_query(OTHER_SENTENCE): [result_with(OTHER_SENTENCE, f"https://e{i}.example") for i in range(11)],
        make_query(SOURCE_EN)[:40]: [SearchResult("only title words", "https://v.example", "age-related macular")],
    }
    notices = []
    hits = web_pass(text, spans(text, WEB_SENTENCE, OTHER_SENTENCE, SOURCE_EN), 40, notices, search=fake_search(answers))
    assert [text[h.q_start:h.q_end] for h in hits] == [WEB_SENTENCE]
    assert [r.url for r in hits[0].results] == ["https://example.org/page"]
    assert hits[0].words >= 8
    assert any("3 文を検索し、1 文が" in n and "一般的な表現" in n for n in notices)


def test_web_pass_uses_cache_and_limits_queries():
    text = f"{WEB_SENTENCE}. {OTHER_SENTENCE}."
    calls = []
    search = fake_search({make_query(WEB_SENTENCE): [result_with(WEB_SENTENCE)]}, calls)
    targets = spans(text, WEB_SENTENCE, OTHER_SENTENCE)
    notices = []
    first = web_pass(text, targets, 40, notices, max_queries=1, search=search)
    assert len(calls) == 1 and len(first) == 1
    assert any("先頭の 1 文" in n for n in notices)
    second = web_pass(text, targets, 40, [], max_queries=1, search=search)
    assert len(calls) == 1 and second[0].results == first[0].results


def test_web_pass_stops_on_invalid_key_and_retries_rate_limit_once():
    text = f"{WEB_SENTENCE}. {OTHER_SENTENCE}."
    targets = spans(text, WEB_SENTENCE, OTHER_SENTENCE)
    calls, notices = [], []
    web_pass(text, targets, 40, notices, search=fake_search({"aqueous": InvalidKey("無効")}, calls))
    assert len(calls) == 1
    assert any("途中で打ち切りました（無効）" in n for n in notices)

    attempts = []

    def flaky(query):
        attempts.append(query)
        if len(attempts) == 1:
            raise RateLimited("上限")
        return [result_with(WEB_SENTENCE)] if "aqueous" in query else []

    hits = web_pass(text, targets, 40, [], search=flaky)
    assert len(attempts) == 3 and len(hits) == 1


def test_web_pass_without_key_sends_nothing(monkeypatch):
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    notices = []
    assert web_pass(WEB_SENTENCE, [(0, len(WEB_SENTENCE))], 40, notices) == []
    assert any("BRAVE_API_KEY" in n for n in notices)


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "index") as s:
        add_doc(s, "pmid:1", SOURCE_EN)
        yield s


def test_check_document_searches_only_unmatched_sentences(store, tmp_path, monkeypatch):
    monkeypatch.setenv("BRAVE_API_KEY", "key")
    calls = []

    def search(client, key, query):
        calls.append(query)
        return [result_with(WEB_SENTENCE, "javascript:alert(1)")] if "aqueous" in query else []

    monkeypatch.setattr(websearch, "brave_search", search)
    path = tmp_path / "manuscript.txt"
    path.write_text(f"{SOURCE_EN}\n\n{WEB_SENTENCE}.\n\n{OTHER_SENTENCE}.", encoding="utf-8")
    report = check_document(path, store, CheckOptions(use_semantic=False, use_web=True))
    assert len(calls) == 2 and not any("macular" in c for c in calls)
    assert [report.text[h.q_start:h.q_end].rstrip('.') for h in report.web_hits] == [WEB_SENTENCE]
    assert report.score > 0  # Web の一致は類似度に含めない
    html = render_html(report)
    assert "Webで見つかった一致" in html and 'class="web"' in html
    assert 'href="javascript:' not in html
    with pymupdf.open("pdf", render_pdf(report, path)) as pdf:
        assert any("Webで見つかった一致" in page.get_text() for page in pdf)


def test_web_search_is_off_by_default(store, tmp_path, monkeypatch):
    monkeypatch.setenv("BRAVE_API_KEY", "key")
    monkeypatch.setattr(websearch, "brave_search", lambda *a: pytest.fail("検索してはいけない"))
    path = tmp_path / "manuscript.txt"
    path.write_text(WEB_SENTENCE, encoding="utf-8")
    report = check_document(path, store, CheckOptions(use_semantic=False))
    assert report.web_hits == []
    assert "Webで一致" not in render_html(report)
