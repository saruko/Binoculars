from bisect import bisect_left
from dataclasses import dataclass, field
from itertools import accumulate
from pathlib import Path

from . import flags as flag_checks
from .align import align, build_query_index
from .citations import check_citations
from .extract import extract_document, sha256_file
from .integrity import check_integrity
from .pdfcheck import check_rendered_text
from .sections import quote_spans, spans_of
from .rerank import DEFAULT_MIN_GAIN, RERANK_MARGIN, get_scorer, source_context
from .semantic import DEFAULT_THRESHOLD, SemanticIndex, is_japanese, sentence_spans, shared_index
from .store import Doc, Store
from .textnorm import SHINGLE_WEIGHT, WORD_WEIGHT, Tokens, shingle_hashes, tokenize, winnow
from .websearch import DEFAULT_MAX_QUERIES, target_sentences, web_pass

NEAR_DUPLICATE_RATIO = 0.8
# この文書数以上に現れる指紋を定型的とみなし、一致の大半が定型的なら定型表現として扱う
COMMON_DF = 10
COMMON_SHARE = 0.8
# 上位の候補とは別に、原稿のこのトークン数ごとに候補を確保する
CANDIDATE_REGION_TOKENS = 100
CANDIDATES_PER_REGION = 20
# 言い換えの候補の出典の文で、許可済みの文書と共有する指紋がこの割合以上なら、許可済みの文書由来とみなす
ALLOWED_SHARE = 0.5
ALLOWED_LABEL = "許可済み文書"


@dataclass(slots=True)
class CheckOptions:
    exclude_references: bool = True
    exclude_quotes: bool = True
    exclude_methods: bool = False
    exclude_preprints: bool = False
    exclude_common: bool = False
    min_words: int = 8
    candidate_rare_df: int = 30
    candidate_region_tokens: int = CANDIDATE_REGION_TOKENS
    candidates_per_region: int = CANDIDATES_PER_REGION
    max_candidates: int = 200
    use_semantic: bool = True
    semantic_threshold: float = DEFAULT_THRESHOLD
    exclude_ext_ids: set[str] = field(default_factory=set)
    # 自分のプレプリントなど、一致してよい文書。共有する部分を確認対象から除く
    allowed_paths: list[str] = field(default_factory=list)
    use_rerank: bool = True
    rerank_min_gain: float = DEFAULT_MIN_GAIN
    # 原稿の文を Brave に送って検索する。利用者が選んだときだけ使う
    use_web: bool = False
    web_max_queries: int = DEFAULT_MAX_QUERIES


@dataclass(slots=True)
class Match:
    q_start: int  # 原稿の文字位置
    q_end: int
    s_start: int  # 出典の文字位置
    s_end: int
    similarity: float
    words: float  # 英単語換算の語数
    token_start: int
    token_end: int


@dataclass(slots=True)
class SourceResult:
    doc: Doc
    matches: list[Match]
    total_weight: int = 0
    unique_weight: int = 0


@dataclass(slots=True)
class SemanticHit:
    q_start: int
    q_end: int
    doc: Doc
    s_start: int
    s_end: int
    score: float


@dataclass(slots=True)
class Report:
    file_name: str
    title: str
    text: str
    score: float
    words_checked: float
    words_excluded: float
    words_matched: float
    sources: list[SourceResult]
    owned_spans: list[tuple[int, int, int]]  # (開始, 終了, sources の添字)
    excluded_spans: list[tuple[int, int, str]]
    semantic_hits: list[SemanticHit]
    flags: list[flag_checks.Flag]
    notices: list[str]
    options: CheckOptions
    index_stats: dict
    triage: list = field(default_factory=list)  # Gemini の仕分け（triage.TriageItem）
    web_hits: list = field(default_factory=list)  # Web照合の一致（websearch.WebHit）


def _exclusions(text: str, options: CheckOptions) -> list[tuple[int, int, str]]:
    spans = []
    if options.exclude_references:
        spans += [(s, e, "参考文献") for s, e in spans_of(text, "references")]
    if options.exclude_methods:
        spans += [(s, e, "方法") for s, e in spans_of(text, "methods")]
    if options.exclude_quotes:
        spans += [(s, e, "引用") for s, e in quote_spans(text)]
    return sorted(spans)


def _scope_tokens(all_tokens: Tokens, spans) -> Tokens:
    excluded = bytearray(len(all_tokens))
    for s, e, _ in spans:
        a = bisect_left(all_tokens.starts, s)
        b = bisect_left(all_tokens.starts, e)
        excluded[a:b] = b"\x01" * (b - a)
    return all_tokens.subset([i for i, x in enumerate(excluded) if not x])


def _load_allowed(paths, store: Store) -> tuple[set[int], set[int], list[str]]:
    """許可済みの文書の指紋と、照合先に登録されている同じファイルの文書IDを返す。"""
    hashes, doc_ids, notices = set(), set(), []
    for path in map(Path, paths):
        found = {h for h, _ in shingle_hashes(tokenize(extract_document(path).text))}
        if not found:
            notices.append(f"許可済みの文書「{path.name}」から文章を抽出できませんでした。スキャンPDFの場合はOCRが必要です。")
        hashes |= found
        doc_id = store.id_for_ext(f"sha256:{sha256_file(path)}")
        if doc_id is not None:
            doc_ids.add(doc_id)
    return hashes, doc_ids, notices


def _subtract(start: int, end: int, spans) -> list[tuple[int, int]]:
    pieces = []
    for s, e, _ in spans:
        if e <= start or s >= end:
            continue
        if s > start:
            pieces.append((start, s))
        start = max(start, e)
    if start < end:
        pieces.append((start, end))
    return pieces


def _allowed_spans(q: Tokens, allowed: set[int], min_weight: int, excluded_spans) -> list[tuple[int, int, str]]:
    """許可済みの文書と最小語数以上つながって一致する範囲。既存の除外範囲とは重ねない。"""
    prefix = [0, *accumulate(q.weights)]
    covered = bytearray(len(q))
    for h, i in shingle_hashes(q):
        if h in allowed:
            j = bisect_left(prefix, prefix[i] + SHINGLE_WEIGHT)
            covered[i:j] = b"" * (j - i)
    spans = []
    i, n = 0, len(q)
    while i < n:
        if not covered[i]:
            i += 1
            continue
        j = i
        while j < n and covered[j]:
            j += 1
        if prefix[j] - prefix[i] >= min_weight:
            spans += [(s, e, ALLOWED_LABEL) for s, e in _subtract(q.starts[i], q.ends[j - 1], excluded_spans)]
        i = j
    return spans


def _from_allowed(text: str, allowed: set[int]) -> bool:
    hashes = [h for h, _ in shingle_hashes(tokenize(text))]
    return bool(hashes) and sum(h in allowed for h in hashes) / len(hashes) >= ALLOWED_SHARE


def _owned_spans(q: Tokens, owner: list[int]) -> list[tuple[int, int, int]]:
    spans = []
    i, n = 0, len(owner)
    while i < n:
        if owner[i] < 0:
            i += 1
            continue
        j = i
        while j + 1 < n and owner[j + 1] == owner[i]:
            j += 1
        spans.append((q.starts[i], q.ends[j], owner[i]))
        i = j + 1
    return spans


def check_document(path: str | Path, store: Store, options: CheckOptions | None = None) -> Report:
    options = options or CheckOptions()
    path = Path(path)
    extracted = extract_document(path)
    text = extracted.text
    notices = []

    excluded_spans = _exclusions(text, options)
    all_tokens = tokenize(text)
    q = _scope_tokens(all_tokens, excluded_spans)
    allowed_hashes, allowed_ids = set(), set()
    if options.allowed_paths:
        allowed_hashes, allowed_ids, allowed_notices = _load_allowed(options.allowed_paths, store)
        notices += allowed_notices
        allowed = _allowed_spans(q, allowed_hashes, options.min_words * WORD_WEIGHT, excluded_spans)
        if allowed:
            before = sum(q.weights)
            excluded_spans = sorted(excluded_spans + allowed)
            q = _scope_tokens(all_tokens, excluded_spans)
            notices.append(f"許可済みの文書と一致する {(before - sum(q.weights)) / WORD_WEIGHT:,.0f} 語を確認対象から除きました。")
        elif allowed_hashes:
            notices.append("許可済みの文書と一致する箇所はありませんでした。")
    total_weight = sum(q.weights)
    all_weight = sum(all_tokens.weights)
    if not total_weight:
        notices.append(
            "すべてが除外範囲に入ったため、確認する文章がありません。" if all_weight
            else "文章を抽出できませんでした。スキャンPDFの場合はOCRが必要です。"
        )

    # 照合先に原稿自身が登録されていると100%一致になるので除外する
    excluded_ids = set()
    self_id = store.id_for_ext(f"sha256:{sha256_file(path)}")
    if self_id is not None:
        excluded_ids.add(self_id)
        notices.append("同じファイルが照合先に登録されていたため、そのファイルは除外しました。")
    if allowed_ids - excluded_ids:
        notices.append("許可済みの文書と同じファイルが照合先に登録されていたため、そのファイルは除外しました。")
    excluded_ids |= allowed_ids
    for ext_id in options.exclude_ext_ids:
        doc_id = store.id_for_ext(ext_id.strip())
        if doc_id is not None:
            excluded_ids.add(doc_id)

    q_shingles = shingle_hashes(q)
    # 照合先の指紋は間引いてあるので、短い一致も拾えるよう原稿側は間引かずに検索する
    hashes = {h for h, _ in q_shingles}
    # 定型表現の判定は、どの文書でも同じものが選ばれる間引いた指紋で数える
    is_common_match = (
        _common_match_test(q_shingles, winnow([h for h, _ in q_shingles]), len(q), store)
        if options.exclude_common else None
    )
    region_of: dict[int, int] = {}
    for h, i in q_shingles:
        region_of.setdefault(h, i // options.candidate_region_tokens)
    candidates = store.lookup(
        hashes, options.max_candidates + len(excluded_ids), region_of=region_of, per_region=options.candidates_per_region,
        rare_df=options.candidate_rare_df,
    )
    candidate_ids = [doc_id for doc_id, _ in candidates if doc_id not in excluded_ids]
    docs = store.get_docs(candidate_ids)

    q_index = build_query_index(q)
    q_sum = [0, *accumulate(q.weights)]
    min_weight = options.min_words * WORD_WEIGHT

    sources: list[SourceResult] = []
    common_tokens = bytearray(len(q))
    for doc in docs.values():
        if options.exclude_preprints and doc.is_preprint:
            continue
        s = tokenize(doc.text)
        kept = []
        for a in align(q, s, q_index):
            if a.weight >= min_weight:
                if is_common_match and is_common_match(a.q_start, a.q_end):
                    common_tokens[a.q_start:a.q_end] = b"\x01" * (a.q_end - a.q_start)
                    continue
                kept.append(Match(
                    q.starts[a.q_start], q.ends[a.q_end - 1], s.starts[a.s_start], s.ends[a.s_end - 1],
                    a.similarity, a.weight / WORD_WEIGHT, a.q_start, a.q_end,
                ))
        if kept:
            sources.append(SourceResult(doc, kept, sum(int(m.words * WORD_WEIGHT) for m in kept)))
    sources.sort(key=lambda r: -r.total_weight)

    # 複数の出典に一致する箇所は、一致が多い出典に帰属させて二重に数えない
    owner = [-1] * len(q)
    for rank, source in enumerate(sources):
        unique = 0
        for m in source.matches:
            for i in range(m.token_start, m.token_end):
                if owner[i] < 0:
                    owner[i] = rank
                    unique += q.weights[i]
        source.unique_weight = unique
    matched_weight = sum(source.unique_weight for source in sources)

    # 表示は寄与の大きい順にする
    order = sorted(range(len(sources)), key=lambda r: -sources[r].unique_weight)
    new_rank = {old: new for new, old in enumerate(order)}
    sources = [sources[old] for old in order]
    owner = [new_rank[o] if o >= 0 else -1 for o in owner]

    # 定型表現は分母からも除く（一致として帰属した部分はそちらを優先）
    common_only = [i for i in range(len(q)) if common_tokens[i] and owner[i] < 0]
    if common_only:
        total_weight -= sum(q.weights[i] for i in common_only)
        for s, e, _ in _owned_spans(q, [0 if common_tokens[i] and owner[i] < 0 else -1 for i in range(len(q))]):
            excluded_spans.append((s, e, "定型表現"))
        excluded_spans.sort()
        notices.append(f"多くの文献に共通する定型表現（ライセンス表記など）を {len(common_only):,} トークン除外しました。")
    score = matched_weight / total_weight if total_weight else 0.0

    if sources and total_weight and sources[0].unique_weight / total_weight >= NEAR_DUPLICATE_RATIO:
        notices.append(
            f"「{sources[0].doc.title}」と原稿の大半が一致しています。同じ文書なら、"
            f"出典ID {sources[0].doc.ext_id} を除外して再度確認してください。"
        )

    semantic_hits = []
    covered = [owner[i] >= 0 or bool(common_tokens[i]) for i in range(len(q))]
    if options.use_semantic:
        if SemanticIndex.exists():
            semantic_hits = _semantic_pass(text, q, covered, store, excluded_ids, options, notices)
            if allowed_hashes:
                # 出典の文が許可済みの文書と同じなら、許可済みの文書を言い換えただけとみなす
                kept = [h for h in semantic_hits if not _from_allowed(h.doc.text[h.s_start:h.s_end], allowed_hashes)]
                if len(kept) < len(semantic_hits):
                    notices.append(f"許可済みの文書に由来する言い換えの候補 {len(semantic_hits) - len(kept)} 件を除きました。")
                semantic_hits = kept
        else:
            notices.append("言い換え検索の索引が未作成のため、言い換えの確認は行っていません。")

    web_hits = []
    if options.use_web:
        # 言い換えの候補になった文は出典が分かっているので、検索の回数を減らすために除く
        targets = target_sentences(
            text, q, covered, options.min_words * WORD_WEIGHT, [(h.q_start, h.q_end) for h in semantic_hits]
        )
        web_hits = web_pass(text, targets, options.min_words * WORD_WEIGHT, notices, options.web_max_queries)

    integrity_flags, integrity_notices = check_integrity(text)
    notices += integrity_notices
    citation_flags, citation_notices = check_citations(text, store, excluded_ids)
    integrity_flags += citation_flags
    notices += citation_notices
    if path.suffix.lower() == ".pdf":
        rendered_flags, rendered_notices = check_rendered_text(path)
        integrity_flags += rendered_flags
        notices += rendered_notices

    return Report(
        file_name=path.name,
        title=extracted.title,
        text=text,
        score=score,
        words_checked=total_weight / WORD_WEIGHT,
        words_excluded=(all_weight - total_weight) / WORD_WEIGHT,
        words_matched=matched_weight / WORD_WEIGHT,
        sources=sources,
        owned_spans=_owned_spans(q, owner),
        excluded_spans=excluded_spans,
        semantic_hits=semantic_hits,
        flags=flag_checks.check_document(path, text) + integrity_flags,
        notices=notices,
        options=options,
        index_stats=store.stats(),
        web_hits=web_hits,
    )


def _common_match_test(q_shingles, selected: set[int], n_tokens: int, store: Store):
    df = store.document_frequencies(selected)
    selected_sum = [0] * (n_tokens + 1)
    common_sum = [0] * (n_tokens + 1)
    for h, i in q_shingles:
        # 一致の境界をまたぐ指紋はどの文書にもないので数えない
        if h in selected and df.get(h, 0) > 0:
            selected_sum[i + 1] = 1
            common_sum[i + 1] = int(df[h] >= COMMON_DF)
    selected_sum = list(accumulate(selected_sum))
    common_sum = list(accumulate(common_sum))

    def test(start: int, end: int) -> bool:
        n = selected_sum[end] - selected_sum[start]
        return n > 0 and (common_sum[end] - common_sum[start]) / n >= COMMON_SHARE

    return test


def _semantic_pass(text, q: Tokens, token_covered, store: Store, excluded_ids, options: CheckOptions, notices: list[str]):
    # すでに完全一致や定型表現として扱った文は対象にしない
    targets = []
    for s, e in sentence_spans(text):
        a = bisect_left(q.starts, s)
        b = bisect_left(q.starts, e)
        if b <= a:
            continue
        if sum(token_covered[a:b]) / (b - a) < 0.5:
            targets.append((s, e))
    if not targets:
        return []

    index = shared_index()
    if index.stale_chunking:
        notices.append("言い換え検索の索引は古い文の分け方で作られているため、出典の長い文の後半は検索していません。index-semantic で作り直すと検索できます。")
    # 再評価するときは候補を広めに取り、話題が同じだけの文は再評価で落とす
    threshold = options.semantic_threshold - RERANK_MARGIN if options.use_rerank else options.semantic_threshold
    found = index.search([text[s:e] for s, e in targets], threshold, excluded_ids)
    pairs = [(t, f) for t, f in zip(targets, found) if f is not None]
    docs = store.get_docs(sorted({f[0] for _, f in pairs}))
    pairs = [(t, f) for t, f in pairs if f[0] in docs]
    if options.use_rerank:
        pairs = _rerank(pairs, text, docs, options, notices)
    return [SemanticHit(s, e, docs[doc_id], ss, se, score) for (s, e), (doc_id, ss, se, score) in pairs]


def _rerank(pairs, text: str, docs, options: CheckOptions, notices: list[str]):
    kept = []
    scorer, load_tried = None, False
    n_failed, first_error = 0, None
    for pair in pairs:
        (s, e), (doc_id, ss, se, score) = pair
        query_ja, source_ja = is_japanese(text[s:e]), is_japanese(docs[doc_id].text[ss:se])
        if query_ja != source_ja:
            # 言語をまたぐ組は別の閾値で選んであり、再評価は英語の文でしか確かめていない
            kept.append(pair)
            continue
        if query_ja:
            if score >= options.semantic_threshold:
                kept.append(pair)
            continue
        gain = None
        try:
            # 読み込みは失敗しても1回だけ試す。文ごとの失敗（GPUのメモリ不足など）は次の文で回復することがある
            if not load_tried:
                load_tried = True
                scorer = get_scorer()
            if scorer is not None:
                gain = scorer.mean_gain(source_context(docs[doc_id].text, ss, se), text[s:e])
        except Exception as e_:
            first_error = first_error or e_
        if gain is None:
            n_failed += 1
            if score >= options.semantic_threshold:
                kept.append(pair)
        elif gain >= options.rerank_min_gain:
            kept.append(pair)
    if n_failed:
        notices.append(f"言い換えの候補 {n_failed} 件は再評価に失敗したので、類似度だけで判定しました: {first_error}")
    return kept
