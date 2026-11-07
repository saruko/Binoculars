"""Web照合: 照合先で一致しなかった文を Brave Search API で完全一致検索し、同じ文があるWebページを探す。

検索結果の題名と抜粋に、原稿の文と同じ語の並びが実際にあるものだけを一致として数える。
原稿の文が外部（Brave）に送られるので、利用者が選んだときだけ使う。
"""
import json
import sqlite3
import time
from bisect import bisect_left
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

from . import DATA_DIR, apikeys
from .semantic import sentence_spans
from .textnorm import WORD_WEIGHT, Tokens, tokenize

KEY_NAME = "BRAVE_API_KEY"
ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
CACHE_PATH = DATA_DIR / "web_cache.sqlite"
CACHE_DAYS = 7
DEFAULT_MAX_QUERIES = 300
# 無料の利用枠は1秒に1回まで
INTERVAL_SECONDS = 1.1
RATE_LIMIT_WAIT = 5.0
# 検索語の長さ（英語30語、日本語75文字ぶん）。長い完全一致検索はほとんど結果を返さない
MAX_QUERY_WEIGHT = 30 * WORD_WEIGHT
# 一致する結果がこれより多い文は、一般的な表現とみなして一致に数えない
COMMON_RESULTS = 10
MAX_ERRORS = 3
KEPT_RESULTS = 5
_QUOTES = str.maketrans({c: " " for c in "\"“”„«»「」『』\n\r\t\f"})


@dataclass(slots=True)
class SearchResult:
    title: str
    url: str
    snippet: str


@dataclass(slots=True)
class WebHit:
    q_start: int
    q_end: int
    query: str
    words: float  # 検索結果と一致した最長の語数（英単語換算）
    results: list[SearchResult] = field(default_factory=list)


class SearchError(Exception):
    fatal = False


class InvalidKey(SearchError):
    fatal = True


class RateLimited(SearchError):
    pass


def brave_search(client: httpx.Client, key: str, query: str) -> list[SearchResult]:
    try:
        response = client.get(
            ENDPOINT,
            params={"q": f'"{query}"', "count": 20, "extra_snippets": "true", "text_decorations": "false"},
            headers={"Accept": "application/json", "X-Subscription-Token": key},
        )
    except httpx.HTTPError as e:
        raise SearchError(f"通信エラー: {e}") from e
    if response.status_code in (401, 403):
        raise InvalidKey("APIキーが無効か、利用の上限に達しています。")
    if response.status_code == 429:
        raise RateLimited("検索回数の上限に達しました。")
    if response.status_code != 200:
        raise SearchError(f"検索サービスのエラー（HTTP {response.status_code}）")
    try:
        return parse_brave(response.json())
    except ValueError as e:
        raise SearchError(f"応答を解析できません: {e}") from e


def parse_brave(data: dict) -> list[SearchResult]:
    results = []
    for item in (data.get("web") or {}).get("results") or []:
        snippet = " … ".join(x for x in [item.get("description") or "", *(item.get("extra_snippets") or [])] if x)
        results.append(SearchResult(item.get("title") or "", item.get("url") or "", snippet))
    return results


def make_query(text: str) -> str:
    """文の先頭から、検索語の長さに収まる部分を取り出す。完全一致検索を壊す引用符と改行は空白にする。"""
    tokens = tokenize(text)
    if not len(tokens):
        return ""
    total, end = 0, 0
    for i, weight in enumerate(tokens.weights):
        if total + weight > MAX_QUERY_WEIGHT:
            break
        total += weight
        end = i
    query = text[tokens.starts[0]:tokens.ends[end]].translate(_QUOTES)
    return " ".join(query.split())


def _longest_run(q: Tokens, r: Tokens) -> int:
    """連続して一致するトークン列の重みの最大値。"""
    best = 0
    previous = [0] * (len(r) + 1)
    for i in range(len(q)):
        current = [0] * (len(r) + 1)
        for j in range(len(r)):
            if q.tokens[i] == r.tokens[j]:
                current[j + 1] = previous[j] + q.weights[i]
                best = max(best, current[j + 1])
        previous = current
    return best


def matched_weight(query: str, result: SearchResult) -> int:
    return _longest_run(tokenize(query), tokenize(f"{result.title} {result.snippet}"))


class _Cache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS search (query TEXT PRIMARY KEY, created REAL, results TEXT)")
        self.db.execute("DELETE FROM search WHERE created < ?", (time.time() - CACHE_DAYS * 86400,))
        self.db.commit()

    def get(self, query: str) -> list[SearchResult] | None:
        row = self.db.execute("SELECT results FROM search WHERE query = ?", (query,)).fetchone()
        return [SearchResult(**x) for x in json.loads(row[0])] if row else None

    def put(self, query: str, results: list[SearchResult]):
        self.db.execute(
            "INSERT OR REPLACE INTO search VALUES (?, ?, ?)",
            (query, time.time(), json.dumps([asdict(r) for r in results], ensure_ascii=False)),
        )
        self.db.commit()

    def close(self):
        self.db.close()


def target_sentences(text: str, q: Tokens, covered, min_weight: int, skip_spans=()) -> list[tuple[int, int]]:
    """照合先との一致や除外範囲で半分以上が覆われていない文。"""
    skip = sorted(skip_spans)
    targets = []
    for s, e in sentence_spans(text):
        a = bisect_left(q.starts, s)
        b = bisect_left(q.starts, e)
        if b <= a or sum(covered[a:b]) / (b - a) >= 0.5:
            continue
        if sum(q.weights[a:b]) < min_weight:
            continue
        if any(ss < e and s < se for ss, se in skip):
            continue
        targets.append((s, e))
    return targets


def web_pass(text: str, targets: list[tuple[int, int]], min_weight: int, notices: list[str],
             max_queries: int = DEFAULT_MAX_QUERIES, search=None, cache_path: Path | None = None,
             sleep=time.sleep) -> list[WebHit]:
    """`search(query)` は検索結果を返す関数。省略すると Brave Search API を使う。"""
    client = None
    if search is None:
        key = apikeys.get(KEY_NAME)
        if not key:
            notices.append(
                f"Web照合には Brave Search API のキー（{KEY_NAME}）が必要です。環境変数か「APIキー」タブで設定してください。"
            )
            return []
        client = httpx.Client(timeout=20)
        search = lambda query: brave_search(client, key, query)
    if not targets:
        notices.append("Web照合: 照合先と一致していない文がないため、検索しませんでした。")
        return []
    if len(targets) > max_queries:
        notices.append(f"Web照合: 対象の {len(targets)} 文のうち、先頭の {max_queries} 文だけを検索しました。")
        targets = targets[:max_queries]

    cache = _Cache(cache_path or CACHE_PATH)
    hits, searched, common, errors = [], 0, 0, 0
    stopped = None
    try:
        for s, e in targets:
            query = make_query(text[s:e])
            if not query:
                continue
            results = cache.get(query)
            if results is None:
                try:
                    sleep(INTERVAL_SECONDS)
                    try:
                        results = search(query)
                    except RateLimited:
                        sleep(RATE_LIMIT_WAIT)
                        results = search(query)
                except SearchError as e_:
                    errors += 1
                    if e_.fatal or isinstance(e_, RateLimited) or errors >= MAX_ERRORS:
                        stopped = str(e_)
                        break
                    continue
                cache.put(query, results)
            searched += 1
            needed = min(min_weight, sum(tokenize(query).weights))
            verified = []
            for result in results:
                weight = matched_weight(query, result)
                if weight >= needed:
                    verified.append((weight, result))
            if len(verified) > COMMON_RESULTS:
                common += 1
            elif verified:
                verified.sort(key=lambda x: -x[0])
                hits.append(WebHit(s, e, query, verified[0][0] / WORD_WEIGHT, [r for _, r in verified[:KEPT_RESULTS]]))
    finally:
        cache.close()
        if client is not None:
            client.close()

    summary = f"Web照合: {searched} 文を検索し、{len(hits)} 文が検索結果と一致しました。"
    if common:
        summary += f"一致する結果が {COMMON_RESULTS} 件を超えた {common} 文は、一般的な表現とみなして挙げていません。"
    notices.append(summary + "検索結果の抜粋だけで判定しているので、見つからなくてもWeb上にないとは限りません。")
    if errors and not stopped:
        notices.append(f"Web照合: {errors} 文は検索に失敗しました。")
    if stopped:
        notices.append(f"Web照合を途中で打ち切りました（{stopped}）。残りの文は検索していません。")
    return hits
