import json
import math
import os
import re
import shutil
import threading
from pathlib import Path

import faiss
import numpy as np

from . import DATA_DIR
from .store import Store
from .textnorm import WORD_WEIGHT, tokenize

MODEL_NAME = "BAAI/bge-m3"
EMBED_DIM = 1024
MAX_CHUNK_CHARS = 500
# モデルは128トークンで打ち切る。日本語の文は約1.5字で1トークンになるので、ほぼすべての文が収まる長さにする
MAX_CHUNK_CHARS_JA = 150
MIN_SENTENCE_WEIGHT = 8 * WORD_WEIGHT
BATCH_SIZE = 128
DEFAULT_THRESHOLD = 0.85
# 言語が異なる文の組は類似度が低く出るので、別の閾値を使う（evaluation/detection/results.md）
CROSS_LANGUAGE_THRESHOLD = 0.75
# 量子化した索引でも1文書あたり約10文で約10KBになるので、件数を絞る
MAX_PUBMED_DOCS = 100_000
CROSS_SEARCH_K = 5
# 文数がこれを超えたら全件を調べず、近い区画だけを調べる索引にする
IVF_MIN_CHUNKS = 2_000_000
IVF_TRAIN_PER_LIST = 39
NPROBE = 128
ENCODE_CHUNKS = 20_000
PART_CHUNKS = 200_000

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(“\"぀-鿿])|(?<=[。！？])|\n\s*\n|\f")
_JAPANESE = re.compile(r"[぀-ヿ㐀-鿿]")
_CHUNK_BREAK = re.compile(r"[、，,;；:：\s]")
_model = None
_shared_index = None
_shared_index_lock = threading.Lock()


def _split_long(text: str, s: int, e: int) -> list[tuple[int, int]]:
    """長い文の後半が検索から漏れないよう、モデルに収まる長さに分ける。重ねないので、同じ箇所を二重に挙げない。"""
    pieces = []
    while True:
        limit = MAX_CHUNK_CHARS_JA if is_japanese(text[s:s + MAX_CHUNK_CHARS]) else MAX_CHUNK_CHARS
        if e - s <= limit:
            pieces.append((s, e))
            return pieces
        # 語や句の途中で切らない。切れ目がなければ長さで切る
        cut = s + limit
        for m in _CHUNK_BREAK.finditer(text, s + limit // 2, s + limit):
            cut = m.end()
        pieces.append((s, cut))
        s = cut


def sentence_spans(text: str) -> list[tuple[int, int]]:
    spans, pos = [], 0
    for m in _SENTENCE_END.finditer(text):
        if m.end() == m.start() and m.start() == pos:
            continue
        spans.append((pos, m.start()))
        pos = m.end()
    spans.append((pos, len(text)))
    result = []
    for span in spans:
        for s, e in _split_long(text, *span):
            if sum(tokenize(text[s:e]).weights) >= MIN_SENTENCE_WEIGHT:
                result.append((s, e))
    return result


def is_japanese(text: str) -> bool:
    return len(_JAPANESE.findall(text)) / max(len(text), 1) > 0.3


def _get_model():
    global _model
    if _model is None:
        import torch
        from sentence_transformers import SentenceTransformer

        # 原稿を外部に出さないよう、ローカルのキャッシュからのみ読み込む
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = SentenceTransformer(MODEL_NAME, device=device, local_files_only=True)
        _model.max_seq_length = 128
        if device == "cuda":
            _model.half()
    return _model


def encode(texts: list[str]) -> np.ndarray:
    return _get_model().encode(
        texts, batch_size=BATCH_SIZE, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    ).astype(np.float32)


def build_index(store: Store, root: str | Path = DATA_DIR / "semantic", progress=print) -> dict:
    """手元のPDFの文単位で埋め込みを作る。PubMed全件は容量と時間の都合で対象外。"""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    doc_ids = store.iter_doc_ids("library")

    chunks = []
    vector_parts = []
    batch_texts: list[str] = []

    def flush():
        if batch_texts:
            vector_parts.append(encode(batch_texts))
            batch_texts.clear()

    for done, i in enumerate(range(0, len(doc_ids), 20), 1):
        docs = store.get_docs(doc_ids[i:i + 20])
        for doc_id, doc in docs.items():
            for s, e in sentence_spans(doc.text):
                chunks.append((doc_id, s, e))
                batch_texts.append(doc.text[s:e])
        if len(batch_texts) >= 4096:
            flush()
            progress(f"  {min(i + 20, len(doc_ids))}/{len(doc_ids)} 文書, {len(chunks)} 文")
    flush()

    vectors = np.concatenate(vector_parts) if vector_parts else np.zeros((0, EMBED_DIM), np.float32)
    np.save(root / "vectors.npy", vectors)
    chunk_array = np.array(chunks, dtype=np.int64).reshape(-1, 3)
    np.save(root / "chunks.npy", chunk_array)
    (root / "info.json").write_text(json.dumps({"model": MODEL_NAME, "chunks": len(chunks)}), encoding="utf-8")
    build_language_map(store, root)
    return {"docs": len(doc_ids), "chunks": len(chunks)}


def build_language_map(store: Store, root: str | Path = DATA_DIR / "semantic") -> dict:
    """手元のPDFの各文が日本語かを記録する。埋め込みは作り直さない。"""
    root = Path(root)
    chunks = np.load(root / "chunks.npy")
    japanese = np.zeros(len(chunks), dtype=bool)
    order = np.argsort(chunks[:, 0], kind="stable")
    for start in range(0, len(order), 5000):
        part = order[start:start + 5000]
        docs = store.get_docs(sorted({int(chunks[i, 0]) for i in part}))
        for i in part:
            doc = docs.get(int(chunks[i, 0]))
            if doc:
                japanese[i] = is_japanese(doc.text[int(chunks[i, 1]):int(chunks[i, 2])])
    np.save(root / "chunks_ja.npy", japanese)
    return {"chunks": len(chunks), "japanese": int(japanese.sum())}


def _add_documents(store: Store, index, doc_ids: list[int], progress) -> list[tuple[int, int, int]]:
    chunks: list[tuple[int, int, int]] = []
    batch_texts: list[str] = []

    def flush():
        if not batch_texts:
            return
        vectors = encode(batch_texts)
        batch_texts.clear()
        if not index.is_trained:
            index.train(vectors)
        index.add(vectors)

    for done, i in enumerate(range(0, len(doc_ids), 500), 1):
        docs = store.get_docs(doc_ids[i:i + 500])
        for doc_id, doc in docs.items():
            for s, e in sentence_spans(doc.text):
                chunks.append((doc_id, s, e))
                batch_texts.append(doc.text[s:e])
        if len(batch_texts) >= 4096:
            flush()
        if done % 20 == 0:
            progress(f"  {min(i + 500, len(doc_ids)):,}/{len(doc_ids):,} 文書, {len(chunks):,} 文")
    flush()
    return chunks


def _write_part(root: Path, index, chunks: np.ndarray):
    # 途中で止まっても元の索引が残るよう、書き終えてから置き換える
    faiss.write_index(index, str(root / "index.faiss.tmp"))
    os.replace(root / "index.faiss.tmp", root / "index.faiss")
    with open(root / "chunks.npy.tmp", "wb") as f:
        np.save(f, chunks)
    os.replace(root / "chunks.npy.tmp", root / "chunks.npy")
    (root / "info.json").write_text(json.dumps({"model": MODEL_NAME, "chunks": len(chunks)}), encoding="utf-8")


def build_pubmed_index(store: Store, root: str | Path = DATA_DIR / "semantic_pubmed", max_docs: int = MAX_PUBMED_DOCS,
                       progress=print) -> dict:
    """PubMed抄録の文単位の索引を作る。全件は容量の都合で扱えないので、先に登録した文献から max_docs 件に限る。"""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(root / "parts", ignore_errors=True)
    doc_ids = store.iter_doc_ids("pubmed")[:max_docs]
    index = faiss.IndexScalarQuantizer(EMBED_DIM, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT)
    chunks = _add_documents(store, index, doc_ids, progress)
    _write_part(root, index, np.array(chunks, dtype=np.int64).reshape(-1, 3))
    return {"docs": len(doc_ids), "chunks": len(chunks)}


def matching_pubmed_docs(store: Store, pattern: str) -> list[int]:
    """題名か雑誌名が正規表現に合うPubMed文献を返す。"""
    matcher = re.compile(pattern, re.IGNORECASE)
    reader = store.conn.cursor()
    with store._lock:
        return [
            doc_id for doc_id, title, journal in reader.execute(
                "SELECT doc_id, title, json_extract(meta, '$.journal') FROM docs WHERE source='pubmed'"
            )
            if matcher.search(f"{title or ''} {journal or ''}")
        ]


def _save_vectors(parts: Path, number: int, chunks, vectors: np.ndarray):
    tmp = parts / f"{number:05d}.tmp"
    with open(tmp, "wb") as f:
        np.savez(f, chunks=np.array(chunks, dtype=np.int64).reshape(-1, 3), vectors=vectors.astype(np.float16))
    os.replace(tmp, parts / f"{number:05d}.npz")


def _encode_to_parts(store: Store, doc_ids: list[int], parts: Path, number: int, progress) -> int:
    """文の埋め込みを小分けに保存する。止まっても、保存済みの文献は次回に作り直さない。"""
    added, chunks, texts, vector_parts = 0, [], [], []

    def encode_pending():
        if texts:
            vector_parts.append(encode(texts))
            texts.clear()

    def save():
        nonlocal number, added, chunks
        encode_pending()
        if chunks:
            number += 1
            _save_vectors(parts, number, chunks, np.concatenate(vector_parts))
            added += len(chunks)
            chunks = []
            vector_parts.clear()

    for done, i in enumerate(range(0, len(doc_ids), 500), 1):
        docs = store.get_docs(doc_ids[i:i + 500])
        for doc_id, doc in docs.items():
            for s, e in sentence_spans(doc.text):
                chunks.append((doc_id, s, e))
                texts.append(doc.text[s:e])
        # 長さの近い文をまとめるほど符号化が速いので、ある程度ためてから渡す
        if len(texts) >= ENCODE_CHUNKS:
            encode_pending()
        if len(chunks) >= PART_CHUNKS:
            save()
        if done % 200 == 0:
            progress(f"  {min(i + 500, len(doc_ids)):,}/{len(doc_ids):,} 文書, {added + len(chunks):,} 文")
    save()
    return added


def _new_index(n_chunks: int):
    if n_chunks < IVF_MIN_CHUNKS:
        return faiss.IndexScalarQuantizer(EMBED_DIM, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT)
    nlist = max(1, min(4 * int(math.sqrt(n_chunks)), n_chunks // IVF_TRAIN_PER_LIST))
    # 差分を符号化しないので、類似度の値は全件を調べる索引と揃う
    index = faiss.IndexIVFScalarQuantizer(
        faiss.IndexFlatIP(EMBED_DIM), EMBED_DIM, nlist, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT, False,
    )
    index.nprobe = min(NPROBE, nlist)
    return index


def _build_from_parts(root: Path, progress) -> int:
    paths = sorted((root / "parts").glob("*.npz"))
    sizes = [np.load(path)["chunks"].shape[0] for path in paths]
    total = sum(sizes)
    index = _new_index(total)
    wanted = index.nlist * IVF_TRAIN_PER_LIST if isinstance(index, faiss.IndexIVF) else 100_000
    rng = np.random.default_rng(0)
    sample = []
    for path, size in zip(paths, sizes):
        count = min(size, max(1, wanted * size // max(total, 1)))
        rows = np.sort(rng.choice(size, count, replace=False))
        sample.append(np.load(path)["vectors"][rows].astype(np.float32))
    progress(f"  学習 {sum(map(len, sample)):,} 文")
    index.train(np.concatenate(sample))
    del sample
    chunks = []
    for done, path in enumerate(paths, 1):
        with np.load(path) as part:
            index.add(part["vectors"].astype(np.float32))
            chunks.append(part["chunks"])
        progress(f"  追加 {done}/{len(paths)}")
    _write_part(root, index, np.concatenate(chunks))
    return total


def extend_index(store: Store, doc_ids: list[int], root: str | Path = DATA_DIR / "semantic_pubmed", progress=print) -> dict:
    """言い換え検索の索引に文献を加える。埋め込みを作り直すのは、まだ保存していない文献だけ。"""
    root = Path(root)
    parts = root / "parts"
    parts.mkdir(exist_ok=True)
    if not any(parts.glob("*.npz")):
        # 今の索引の中身を最初の部分にする。量子化した値から戻すので、符号化し直すより精度はわずかに落ちる
        index = faiss.read_index(str(root / "index.faiss"))
        _save_vectors(parts, 0, np.load(root / "chunks.npy"), index.reconstruct_n(0, index.ntotal))
    paths = sorted(parts.glob("*.npz"))
    known = set()
    for path in paths:
        known.update(np.unique(np.load(path)["chunks"][:, 0]).tolist())
    new_ids = [doc_id for doc_id in doc_ids if doc_id not in known]
    progress(f"  追加 {len(new_ids):,} 文書（保存済みの {len(doc_ids) - len(new_ids):,} 文書を除く）")
    added = _encode_to_parts(store, new_ids, parts, int(paths[-1].stem), progress)
    return {"docs": len(new_ids), "chunks": added, "total_chunks": _build_from_parts(root, progress)}


class SemanticIndex:
    def __init__(self, root: str | Path = DATA_DIR / "semantic", pubmed_root: str | Path = DATA_DIR / "semantic_pubmed"):
        root, pubmed_root = Path(root), Path(pubmed_root)
        self.chunks = np.load(root / "chunks.npy")
        vectors = np.load(root / "vectors.npy")
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        # (索引, 文の位置) の組。手元のPDFとPubMedで索引が分かれる
        self.parts = [(self.index, self.chunks)]
        if (pubmed_root / "index.faiss").exists():
            self.parts.append((faiss.read_index(str(pubmed_root / "index.faiss")), np.load(pubmed_root / "chunks.npy")))
        # 英語の文と日本語の文の組だけ別の閾値で探す
        self.ja_index, self.ja_chunks = None, None
        if (root / "chunks_ja.npy").exists():
            japanese = np.flatnonzero(np.load(root / "chunks_ja.npy"))
            if japanese.size:
                self.ja_index = faiss.IndexFlatIP(vectors.shape[1])
                self.ja_index.add(vectors[japanese])
                self.ja_chunks = self.chunks[japanese]

    @staticmethod
    def exists(root: str | Path = DATA_DIR / "semantic") -> bool:
        return (Path(root) / "vectors.npy").exists()

    def search(self, texts: list[str], threshold: float, exclude_doc_ids: set[int], k: int = 5,
               cross_threshold: float | None = CROSS_LANGUAGE_THRESHOLD):
        """各クエリ文について、閾値以上で最も近い (doc_id, start, end, score) か None を返す。"""
        if not texts:
            return []
        vectors = encode(texts)
        results = [None] * len(texts)

        def consider(row, index, chunks, minimum, top_k, keep_existing=False):
            scores, ids = index.search(vectors[row], top_k)
            for q, row_scores, row_ids in zip(row, scores, ids):
                if keep_existing and results[q] is not None:
                    continue
                for score, idx in zip(row_scores, row_ids):
                    if idx < 0 or score < minimum:
                        break
                    doc_id, s, e = chunks[idx]
                    if int(doc_id) not in exclude_doc_ids:
                        if results[q] is None or score > results[q][3]:
                            results[q] = (int(doc_id), int(s), int(e), float(score))
                        break

        everything = np.arange(len(texts))
        for index, chunks in self.parts:
            consider(everything, index, chunks, threshold, k)
        if cross_threshold is not None and self.ja_index is not None:
            english = np.array([i for i, text in enumerate(texts) if not is_japanese(text)], dtype=np.int64)
            if english.size:
                consider(english, self.ja_index, self.ja_chunks, cross_threshold, CROSS_SEARCH_K, keep_existing=True)
        return results


def _index_signature(root: Path, pubmed_root: Path) -> list:
    files = [root / "vectors.npy", root / "chunks.npy", root / "chunks_ja.npy",
             pubmed_root / "index.faiss", pubmed_root / "chunks.npy"]
    return [(path, path.stat().st_mtime_ns if path.exists() else None) for path in files]


def shared_index(root: str | Path = DATA_DIR / "semantic", pubmed_root: str | Path = DATA_DIR / "semantic_pubmed") -> SemanticIndex:
    """読み込み済みの索引を返す。大きくて読み込みに時間がかかるので、ファイルが変わるまで使い回す。"""
    global _shared_index
    root, pubmed_root = Path(root), Path(pubmed_root)
    signature = _index_signature(root, pubmed_root)
    with _shared_index_lock:
        if _shared_index is None or _shared_index[0] != signature:
            # 新旧の索引が同時にメモリに載らないよう、先に手放す
            _shared_index = None
            _shared_index = (signature, SemanticIndex(root, pubmed_root))
        return _shared_index[1]
