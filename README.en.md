# Binoculars Pre-submission Checker

[日本語](README.md) | English

This tool adds features to [Binoculars](https://github.com/ahans30/Binoculars) (Hans et al., ICLR 2024) for checking a manuscript on your own computer before submission. It is not an official release from the original developers.

The web UI and the documents linked from this page are in Japanese.

## Differences from the original Binoculars
The original Binoculars implements a score for detecting AI-generated text. This repository reimplements that score and adds similarity checking and other features.

| Item | Original Binoculars | This repository |
|---|---|---|
| AI-generated text score | Uses Falcon-7B by default and classifies text as AI-generated or human-written with a fixed threshold | Same calculation (log PPL / log Cross-PPL), implemented in `app.py`. You can choose the models, and long documents are split every 512 tokens so that the whole text is evaluated. **It gives no classification** (see [performance on medical papers](#performance-on-medical-papers)) |
| Similarity check | None | Added (fingerprint matching, PDF and HTML reports) |
| Paraphrase candidates | None | Added (sentence embeddings and reranking) |
| Flags | None | Added (tortured phrases, citations of retracted papers, character manipulation, rendered-versus-extracted text mismatch, citation overlap) |
| Revision suggestions | None | Added (uses an LLM on your own computer) |
| Detection evaluation | See the paper | Added (methods and results in `evaluation/`) |
| Interface | Gradio demo | Web UI with three tabs, and GitHub Actions workflows |

The original code (`binoculars/`, `datasets/`, `experiments/`, `demo/`, `main.py`, `setup.py`) is kept unchanged. This tool does not use it, and it has not been tested in this repository. `app.py` and `requirements.txt` have been replaced, so follow the instructions in the [original repository](https://github.com/ahans30/Binoculars) if you want to run the original code.

## Features
The web UI has three tabs.

- **Similarity check**: Compares a document with PubMed abstracts, PMC full texts (CC-licensed only), and your own PDFs, and reports the matching passages.
- **Revision suggestions**: For each matched sentence, an LLM on your computer suggests how to handle it and drafts a revision.
- **Binoculars score**: Calculates a score for a text with the method from the ICLR 2024 paper *"Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text"*. It does not determine whether AI was used.

To run the similarity check on GitHub Actions, see the [guide](docs/github-actions.md). **This repository is public, so do not put manuscripts in it.** To use Actions, create a separate private repository.

## Getting started
1. Double-click `start.bat`.
2. On the first run, it creates a virtual environment (Python 3.14) and installs the libraries, including PyTorch for CUDA 12.8.
3. When setup finishes, your browser opens `http://127.0.0.1:7860`.

`start.bat` installs PyTorch first from the CUDA 12.8 index (`--index-url`). `requirements.txt` does not name a PyTorch index, because with `--extra-index-url` pip installs whichever version is higher, including a package of the same name on PyPI.

Known unfixed vulnerability: gradio (including the latest 6.31.0) requires `pillow<13.0`, so Pillow stays on 12.x and cannot move to the fixed 13.0.0 (this includes a heap overflow in the bundled libopenjp2). Until gradio supports Pillow 13, do not load untrusted images or PDFs.

## Similarity check

### Registering sources
Run the commands with the Python in the virtual environment (`venv\Scripts\python.exe -m simcheck ...`).

| Command | Description |
|---|---|
| `add-library "folder"` | Registers the PDFs in a folder. The original files are not written to. Use `--rebuild` to register them again |
| `index-semantic` | Builds the paraphrase search index for your PDFs. Run it again after registering the PDFs again |
| `index-semantic --pubmed` | Builds the paraphrase search index for PubMed abstracts. Set the number of documents with `--max-docs` (the default is 100,000, taken from the documents registered first) |
| `index-semantic --add-matching "regex"` | Adds PubMed abstracts whose title or journal name matches to the paraphrase search index. Build the index with `--pubmed` first |
| `index-semantic --add-pmc` | Adds registered PMC full texts to the paraphrase search index |
| `index-citations` | Builds the list used for the citation overlap check. Run it again after adding documents |
| `add-pubmed-query "query"` | Registers PubMed abstracts that match a query (up to 10,000) |
| `add-pmc-query "query"` | Registers CC-licensed PMC full texts that match a query. Queries with more than 10,000 results are retrieved in parts by publication month |
| `build-corpus file` | Registers everything from a file of queries (see `corpus/`) |
| `add-pubmed` | Registers the PubMed baseline. Use `--first` and `--last` to choose file numbers. It resumes after an interruption, and it stops when free disk space falls below 40 GB |
| `update-integrity` | Downloads the tortured phrase list and the retraction data. The retraction data is updated every weekday, so run this before a check |
| `compact` | Merges the fingerprint index into one. Run it after registering large parts of the baseline |
| `stats` | Shows the state of the index |
| `check "document"` | Checks a document and creates PDF and HTML reports |

Indexes and reports are saved in `data/simcheck/`. The index includes the text of subscription PDFs, so do not share it.

The full baseline is a download of about 52 GB. In an actual run, at about 21 million PubMed abstracts and 140,000 PMC full texts, the document store was about 31 GB and the fingerprints about 16 GB (before merging). The run took more than 7 hours with 6 connections, including retries after connection failures. Search speed on the full baseline has not been checked.

### How detection works
1. Extracts the text from a PDF, DOCX, or TXT file and determines the ranges to exclude, such as references and quotations.
2. Normalizes characters (full-width and half-width forms, ligatures, line-end hyphens, invisible characters, Cyrillic and Armenian letters that look like Latin ones, Greek letters mixed into Latin words, and so on), then splits English into words and Japanese into characters.
3. Finds candidate documents with fingerprints (winnowing), then aligns tokens to locate the matches. Candidates are the top documents overall plus the top documents for every 100 words of the manuscript. A continuous match of 10 or more words always leaves a shared fingerprint. Matches of 8 or 9 words may be missed.
4. A passage that matches several sources is counted once. If you choose to exclude common expressions, expressions shared by many documents are removed from both the matches and the denominator (they are not excluded by default).
5. Paraphrases are found with BAAI/bge-m3 embeddings. English sentences are then narrowed down by how much easier the sentence becomes to predict after reading the source (SCDG, Qwen2.5-1.5B). The search covers your PDFs and the PubMed abstracts that have been indexed. The full PubMed baseline cannot be covered because of the index size. Japanese PDFs are also searched for English sentences. Paraphrases are not included in the similarity value.

### Flags
None of these are included in the similarity value.

- **Tortured phrases**: Compares the text with the [Problematic Paper Screener](https://dbrech.irit.fr/pls/apex/f?p=9999:5) list of phrases known to be traces of paraphrasing tools. References are not checked. Two or more different phrases raise a flag, and a single phrase only produces a note.
- **Citations of retracted papers**: Compares the references with [Retraction Watch](https://gitlab.com/crossref/retraction-watch-data) data by DOI, PMID, and title. Retractions and expressions of concern are shown separately. Papers with only a correction and papers whose retraction was reversed are not shown. Title-only matches are listed separately.
- **Character manipulation**: Shows invisible characters, large numbers of soft hyphens, look-alike characters mixed into Latin words, transparent text, and so on.
- **Rendered-versus-extracted text mismatch**: Shows pages where the appearance of the PDF differs greatly from the extracted text. Up to 4 pages are rendered, read with OCR (RapidOCR) on your computer, and compared.
- **Citation overlap**: Shows documents whose references overlap heavily with those of the checked document. It covers documents processed by `index-citations`.

The detection rates of tortured phrases, rendered-versus-extracted text mismatch, and citation overlap have not been measured. No match does not prove that there is no problem. For the false positive checks of tortured phrases and retractions, see the [evaluation results](evaluation/integrity/results.md).

### Detection evaluation
Detection was measured for each type of alteration, using synthetic manuscripts with reused text embedded in them. Verbatim reuse of 10 or more words is almost always detected, but editing a word every 5 words leaves no fingerprints and is rarely detected. The paraphrase search lists 89–98% of heavily paraphrased passages as candidates, but it also produces 1–2 candidates per 1,000 words in text with no reuse. These values come from synthetic manuscripts and are not detection rates for real manuscripts. For details, see the [evaluation results](evaluation/detection/results.md).

### Reports
- **PDF**: Contains a cover page, summary, source list, the manuscript pages, and paraphrase candidates, in that order. On the manuscript pages, matches are red, excluded ranges are gray, and paraphrase candidates have a dotted underline. Source numbers appear in the left margin. The text of the sources is not included.
- **HTML**: Shows the manuscript and the source text side by side. It includes the text of subscription PDFs, so do not share it.
- DOCX files are converted to PDF with Word to create the manuscript pages. A working copy is used, so the original file is not changed. If Word is not available, only the body text is laid out again, so pages and lines do not match the manuscript.

### Reading the results
- The similarity value is the number of matched words divided by the number of words checked. For Japanese, 2.5 characters count as 1 word.
- **Matches with documents outside the sources (subscription articles, web pages, and so on) cannot be detected.** A low value does not prove that there is no plagiarism.
- Commercial services use different databases and calculations, so their values cannot be compared with these. The result from a journal's check may differ.
- Stock phrases alone can produce a value when no text was reused. With newly written AI text as a control, long texts (844 words on average) had a mean of 1.3% and a maximum of 3.5%, and abstract-length texts had a mean of 5.0% and a maximum of 32.7%. In short texts, one matched sentence can change the value greatly, so check the matched passages rather than relying on the value alone.
- The default minimum match length is 8 words. At 7 words or fewer, matches on stock phrases increase sharply.
- Paraphrases and flags are candidates to review. They are not evidence of misconduct.
  - The default paraphrase threshold is `0.85`. With reranking, candidates with a similarity of 0.82 or higher are narrowed down by gain. If the reranking model cannot be loaded, only the 0.85 similarity threshold is used, and the report says so.
  - Lowering the threshold finds more, but in real papers it also lists many unrelated sentences from the same field.

### Data handling
- Checks run entirely on your computer, and manuscript content is not sent anywhere. External connections are made only to download PubMed and PMC data, the phrase list, and the retraction data. None of them include manuscript content.
- The phrase list has no stated license. Ask the provider before distributing the data in `data/integrity/` to others.
- Documents uploaded in the web UI are copied by Gradio to a temporary folder on your computer.
- Scanned PDFs cannot be registered as sources (OCR is used only for the rendered-versus-extracted text check).

### Tests
```bash
venv/Scripts/python.exe -m pytest tests
```

## Building the sources
The source data is not distributed (see [Data not included](#data-not-included)). Build it on your own computer as follows.

```bash
venv\Scripts\python.exe -m simcheck add-pubmed
venv\Scripts\python.exe -m simcheck build-corpus corpus/queries.txt
venv\Scripts\python.exe -m simcheck add-library "PDF folder"
venv\Scripts\python.exe -m simcheck update-integrity
venv\Scripts\python.exe -m simcheck compact
venv\Scripts\python.exe -m simcheck index-citations
venv\Scripts\python.exe -m simcheck index-semantic
venv\Scripts\python.exe -m simcheck index-semantic --pubmed
```

If you do not need everything, narrow the range by using `add-pubmed --first N --last N` or `build-corpus` instead of `add-pubmed`. The queries in `corpus/queries.txt` are examples, so replace them with queries for your own field. The values in `evaluation/` apply to the sources used at the time of measurement.

## Data not included
`data/` is not included in the repository.

| Data | Why it is not included |
|---|---|
| Text of your PDFs (part of `data/simcheck/docs.sqlite`) | This is the text of subscription articles. Extracting it as text does not remove the copyright |
| PubMed abstracts | Distribution is subject to the [NLM terms](https://www.nlm.nih.gov/databases/download/terms_and_conditions.html) (acknowledging the source, and keeping the data current or stating that it is not), and the copyright of an abstract may belong to the publisher or authors. You can download them yourself |
| PMC full texts | CC license conditions differ from article to article. You can download them yourself |
| Evaluation data (`data/detection_eval/`, `data/ai_threshold/`) | Contains excerpts from subscription PDFs |
| Tortured phrase list (`data/integrity/`) | Has no stated license |
| Fingerprint and embedding indexes | Matches cannot be shown without the text. They total tens of gigabytes and exceed GitHub's file size limit |

PubMed data is provided by the U.S. National Library of Medicine. This repository is not endorsed by NLM.

## Revision suggestions
For each matched sentence, the tool suggests how to handle it (add a citation, quote and cite, or rewrite) and drafts a revision. It indicates a possible stock phrase only when the sentence contains no numbers and the same expression appears in many documents in the sources. This decision is not left to the LLM, because the LLM sometimes answered "no citation needed" even for sentences containing numerical results. The text is not changed automatically. A copy of the Word file, with the author's chosen actions added as comments, is saved in `data/simcheck/revisions/`.

**A rewritten sentence still needs a citation if it uses the content of the source.** Do not aim simply to lower the similarity value, and review each item. Adopted sentences count as AI-assisted editing, so declare them according to the journal's rules.

### How to use
1. Load a model in LM Studio and start the server. The intended local LLMs are Gemma 4, Qwen3.5, DeepSeek, and similar models. Quality has been checked only for the model described in the [evaluation results](evaluation/revise/results.md). The manuscript and source text are passed to this model, so do not connect to an external API.
2. In the revision suggestions tab (「修正案」), select a DOCX file and create the suggestions.
3. Review the suggestions one sentence at a time, edit them if needed, and choose an action.
4. Use the export button (「Wordに書き出す」) to create a copy with comments. The original file is not changed.

### Checks on suggestions
Instead of relying on the LLM's own report, each suggestion is checked on the following points. Suggestions that fail are still shown, but they are placed last with the problems stated.

- Numbers and reference numbers are the same as in the original sentence
- Abbreviations and terms have not been lost
- The number of negative expressions has not changed
- There are no words for an increase, decrease, or direction of comparison that the original sentence does not have
- Expressions of statistical significance have not changed
- The language is the same as the original sentence (suggestions in another language are not shown)
- The suggestion is not a completely different sentence (bge-m3 similarity, threshold 0.75)

The number of words that still match the sources is also shown for reference, but it is not used to order the suggestions.

**Even suggestions that pass the checks cannot be adopted as they are.** In a check with sentences from published abstracts, the meaning clearly changed in 4 of 45 suggestions that passed (for example, `One group` became `experimental group`). The checks cannot detect added facts or replaced methods or subjects, so the author must always review the content before adopting a suggestion. For details, see the [evaluation results](evaluation/revise/results.md). Only exact-match sentences are covered, not paraphrase candidates.

## Binoculars score

### Choosing models
- **Requirement**: The Observer (Base) and Performer (Instruct) must be from **the same family with the same vocabulary (tokenizer)** (for example, `Qwen/Qwen2.5-1.5B` and `Qwen/Qwen2.5-1.5B-Instruct`). Different vocabularies cause an error at load time.
- Load models by local path or Hugging Face model ID.
- A local path must be in Hugging Face format (a folder containing `config.json` and `*.safetensors`). LM Studio GGUF files cannot be loaded.

### Reading the score
- The score is log PPL divided by log Cross-PPL. Lower scores tend to be closer to AI-generated text.
- No "AI-generated" or "human-written" classification or threshold is shown, because the evaluation found no threshold that gives a usable classification.
- Long documents are split every 512 tokens so that the whole text is evaluated, and the score of each segment is also shown.

### Performance on medical papers
**Do not use this score as evidence of whether AI was used.** The Qwen2.5-1.5B pair was evaluated on English medical papers. It did not meet the criteria set in advance, and adjusting the threshold did not solve this.

| Item | Result |
|---|---|
| AUC (newly written AI text, abstract length) | 0.729 (95% confidence interval 0.683–0.774) |
| With the paper's threshold of 0.9015 (for Falcon) | 5.0% of human abstracts misclassified as AI, and 22.6% of AI text detected |
| Misclassification of Japanese affiliations at a threshold of 0.916 (false positive rate of 5% in calibration) | 10.0% (3.3–5.0% for other affiliations; the counts are small, so the difference is not established) |
| AUC for text polished by AI | 0.541 (almost no discrimination) |
| AUC for long text | 0.497 (no discrimination) |

A high score is not evidence that AI was not used, and a low score is not evidence that AI was used. Japanese text has not been evaluated. For details, see the [evaluation results](evaluation/ai_threshold/results.md).

### Display
- The right side of the screen uses color to show how closely each token followed the model's prediction. Green means as predicted, and red means an unexpected word. This does not indicate whether the text came from AI or a human.

## Citation
If you use the method behind the Binoculars score, cite the original paper.

```bibtex
@misc{hans2024spotting,
      title={Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text},
      author={Abhimanyu Hans and Avi Schwarzschild and Valeriia Cherepanova and Hamid Kazemi and Aniruddha Saha and Micah Goldblum and Jonas Geiping and Tom Goldstein},
      year={2024},
      eprint={2401.12070},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## License
The code, both the original Binoculars and the additions, is under the BSD 3-Clause License ([LICENSE.md](LICENSE.md)). The source data is subject to the terms of each provider.
