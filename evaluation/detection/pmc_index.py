"""PMC全文を言い換え索引に加えたときの効果を、今の索引を残したまま dev で測る。

今の標本はPubMed抄録と手元のPDFから作っているので、PMC本文の言い換えを見つけられるかは別の標本で測る。

    venv/Scripts/python.exe evaluation/detection/pmc_index.py sample
    venv/Scripts/python.exe evaluation/detection/pmc_index.py paraphrase
    venv/Scripts/python.exe evaluation/detection/pmc_index.py score --root data/simcheck/semantic_pubmed --tag current
    venv/Scripts/python.exe evaluation/detection/pmc_index.py build --root data/simcheck/semantic_pubmed_pmc
    venv/Scripts/python.exe evaluation/detection/pmc_index.py score --root data/simcheck/semantic_pubmed_pmc --tag pmc
    venv/Scripts/python.exe evaluation/detection/pmc_index.py evaluate --root data/simcheck/semantic_pubmed_pmc --tag pmc --only "llm_*"
"""
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paraphrase
import run
from cases import DATA, Condition, build_cases, english_passage, split_of, stable_rng, summarize
from simcheck import DATA_DIR, engine
from simcheck.engine import CheckOptions
from simcheck.semantic import extend_index, shared_index
from simcheck.store import Store

SPLIT = "dev"
N_PMC = 100
MODES = ("light", "heavy")
CASES_PATH = DATA / f"cases_pmc_{SPLIT}.json"
PARAPHRASES_PATH = DATA / f"paraphrases_pmc_{SPLIT}.jsonl"
BASE_ROOT = DATA_DIR / "semantic_pubmed"


def sample():
    """標本は一度作ったら作り直さない。test に割り当てた論文は使わない。"""
    if CASES_PATH.exists():
        print(f"作成済み: {CASES_PATH}")
        return
    with Store() as store:
        ids = store.iter_doc_ids("pmc")
        rng = stable_rng("cases", "pmc", SPLIT)
        passages = []
        for doc_id in rng.sample(ids, len(ids)):
            doc = store.get_docs([doc_id]).get(doc_id)
            if doc and split_of(doc.ext_id) == SPLIT:
                text = english_passage(doc.text, rng)
                if text:
                    passages.append({"ext_id": doc.ext_id, "text": text})
            if len(passages) >= N_PMC:
                break
    fillers = json.loads((DATA / f"cases_{SPLIT}.json").read_text(encoding="utf-8"))["fillers"]
    CASES_PATH.write_text(json.dumps({"split": SPLIT, "passages": {"pmc": passages}, "fillers": fillers}, ensure_ascii=False), encoding="utf-8")
    print(f"{len(passages)} 件 -> {CASES_PATH}")


def generate():
    # 今の標本と同じモデルで作る。モデル名は公開するファイルに書かない
    with open(DATA / f"paraphrases_{SPLIT}.jsonl", encoding="utf-8") as f:
        paraphrase.MODELS = list(dict.fromkeys(json.loads(line)["model"] for line in f))
    paraphrase.output_path = lambda split: PARAPHRASES_PATH
    paraphrase.prepare_cases = lambda store, split: json.loads(CASES_PATH.read_text(encoding="utf-8"))
    paraphrase.JOBS = [("pmc", mode) for mode in MODES]
    paraphrase.N_PER_CELL = N_PMC
    sys.argv = [sys.argv[0], "--split", SPLIT]
    paraphrase.main()


def build(root: Path):
    """今の索引の部分をそのまま使う。書き換えないファイルなのでハードリンクにして容量を節約する。"""
    parts = root / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    for path in sorted((BASE_ROOT / "parts").glob("*.npz")):
        if not (parts / path.name).exists():
            os.link(path, parts / path.name)
    started = time.time()
    with Store() as store:
        result = extend_index(store, store.iter_doc_ids("pmc"), root=root)
    print(json.dumps({**result, "hours": round((time.time() - started) / 3600, 2)}, ensure_ascii=False), flush=True)


def _use_index(root: Path):
    engine.shared_index = lambda: shared_index(DATA_DIR / "semantic", root)


def score(root: Path, tag: str):
    _use_index(root)
    data = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    texts = {}
    with open(PARAPHRASES_PATH, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            texts[(row["passage_key"], row["mode"])] = row["text"]
    cells = {}
    with Store() as store, tempfile.TemporaryDirectory(dir=DATA) as tmp_dir:
        for mode in MODES:
            condition = Condition(f"llm_{mode}", "LLMの書き換え", ("pmc",), llm_mode=mode, semantic=True)
            cases = build_cases(condition, "pmc", data, texts)
            started = time.time()
            scored = [run.run_case(case, store, CheckOptions(use_semantic=True), tmp_dir) for case in cases]
            summary = summarize(scored, (time.time() - started) / len(cases))
            summary["semantic_any_rate"] = sum(t["semantic_any"] for doc in scored for t in doc["trials"]) / summary["n"]
            cells[f"llm_{mode}/pmc"] = summary
            print(
                f"llm_{mode}/pmc: n={summary['n']} 言い換え={summary['semantic_rate']:.3f} "
                f"地の文の候補={summary['filler_semantic_per_1000']:.2f}/1000語 {summary['seconds_per_doc']:.1f}秒/件",
                flush=True,
            )
    (HERE / f"results_pmc_{SPLIT}_{tag}.json").write_text(json.dumps({
        "root": root.name, "code_hash": run.code_hash(), "finished_at": time.strftime("%Y-%m-%d %H:%M"), "cells": cells,
    }, ensure_ascii=False, indent=1), encoding="utf-8")


def evaluate(root: Path, tag: str, only: str):
    """今の標本で測り、results_dev_full と比べられる形で書き出す。"""
    _use_index(root)
    sys.argv = [sys.argv[0], "--split", SPLIT, "--only", only, "--no-retraction", "--tag", tag]
    run.main()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["sample", "paraphrase", "build", "score", "evaluate"])
    parser.add_argument("--root", type=Path)
    parser.add_argument("--tag")
    parser.add_argument("--only", default="llm_*")
    args = parser.parse_args()
    root = args.root.resolve() if args.root else None
    if args.command == "sample":
        sample()
    elif args.command == "paraphrase":
        generate()
    elif args.command == "build":
        build(root)
    elif args.command == "score":
        score(root, args.tag)
    else:
        evaluate(root, args.tag, args.only)


if __name__ == "__main__":
    main()
