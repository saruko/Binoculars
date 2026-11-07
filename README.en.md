# Binoculars Pre-submission Checker

[日本語](README.md) | English

This tool checks a manuscript on your own computer before submission. It checks text similarity, drafts revision suggestions, and calculates the Binoculars score. It adds features to [Binoculars](https://github.com/ahans30/Binoculars) (Hans et al., ICLR 2024) and is not an official release from the original developers.

The web UI and the documents linked from this page are in Japanese.

**What to know before using it**
- It does not replace a commercial similarity check. Matches with documents outside the sources (subscription articles, web pages, and so on) cannot be detected, and the values cannot be compared.
- The similarity value alone is not enough to judge a manuscript. Text with no reuse also gets a value, so check the matched passages (see [Reading the results](#reading-the-results)).
- Paraphrase candidates and flags are candidates to review. They are not evidence of misconduct.
- Do not use the Binoculars score as evidence of whether AI was used (see [Performance on medical papers](#performance-on-medical-papers)).

## Features

| Feature | Description | Where to use it |
|---|---|---|
| Similarity check | Compares a document with PubMed abstracts, PMC full texts (CC-licensed only), and your own PDFs, and reports the matches in PDF and HTML. Optionally also searches the web | Similarity tab (「類似度チェック」), `check` command, GitHub Actions |
| Paraphrase candidates | Lists sentences that may be reworded, using sentence embeddings and reranking. They are not included in the similarity value | Similarity tab, `check` command |
| Flags | Shows tortured phrases, citations of retracted papers, character manipulation, rendered-versus-extracted text mismatch, and citation overlap | Same as above |
| Revision suggestions | An LLM suggests how to handle each matched sentence and drafts revisions, which are then checked mechanically. By default, the LLM runs on your computer | Revision tab (「修正案」) |
| Binoculars score | Calculates a score for a text. It does not determine whether AI was used | Score tab (「Binocularsスコア」) |

Manuscript content leaves your computer only if you choose an external API (including the web search) or GitHub Actions (see [Data handling](#data-handling)).

## Requirements
- Windows and Python 3.14. `start.bat` installs PyTorch for CUDA 13.2.
- Word is used to create the manuscript pages for DOCX, DOC, RTF and ODT files and to read DOC, RTF and ODT files. Without Word, LibreOffice is used. Without either, DOC, RTF and ODT files cannot be read, and [pages and lines of DOCX files will not match the manuscript](#reports).
- The full sources need about 100 GB of disk space. The baseline download takes about another 52 GB.

The measurements in this repository were made on a PC with 64 GB of memory and 16 GB of GPU memory (GeForce RTX 5080).

## Getting started
1. Double-click `start.bat`.
2. On the first run, it creates a virtual environment and installs the libraries.
3. When setup finishes, your browser opens `http://127.0.0.1:7860`.

The similarity check needs sources, so build them first (see [Building the sources](#building-the-sources)).

## Building the sources
The source data is not distributed (see [Data not included](#data-not-included)). Build it on your own computer. Run the commands with the Python in the virtual environment.

### 1. Download the models
To keep manuscripts from leaving your computer, the paraphrase search loads its models only from the local cache. Download them first.

```bash
venv\Scripts\hf.exe download BAAI/bge-m3
```

```bash
venv\Scripts\hf.exe download Qwen/Qwen2.5-1.5B
```

### 2. Register documents and build the indexes

```bash
venv\Scripts\python.exe -m simcheck add-pubmed
venv\Scripts\python.exe -m simcheck build-corpus corpus/queries.txt
venv\Scripts\python.exe -m simcheck add-library "PDF folder"
venv\Scripts\python.exe -m simcheck update-integrity
venv\Scripts\python.exe -m simcheck compact
venv\Scripts\python.exe -m simcheck index-citations
venv\Scripts\python.exe -m simcheck index-semantic
venv\Scripts\python.exe -m simcheck index-semantic --pubmed
venv\Scripts\python.exe -m simcheck index-semantic --add-matching "regex for your field"
```

- If you do not need everything, narrow the range by using `add-pubmed --first N --last N` or `build-corpus` instead of `add-pubmed`.
- The queries in `corpus/queries.txt` are examples, so replace them with queries for your own field.
- `--add-matching` chooses which fields to add to the paraphrase search, by title or journal name. The evaluation below used ophthalmology and allergy terms (`ophthalm|ocular|\beye|retina|retinal|retinopath|cornea|corneal|glaucoma|cataract|uveitis|macula|myopi|conjunctiv|keratitis|keratoconus|vitre|optic nerve|allergic rhinitis|pollinosis|hay fever`).
- The values in `evaluation/` apply to the sources used at the time of measurement.

### Commands

| Command | Description |
|---|---|
| `add-pubmed` | Registers the PubMed baseline. Use `--first` and `--last` to choose file numbers. It resumes after an interruption, and it stops when free disk space falls below 40 GB |
| `add-pubmed-query "query"` | Registers PubMed abstracts that match a query (up to 10,000) |
| `add-pmc-query "query"` | Registers CC-licensed PMC full texts that match a query. Queries with more than 10,000 results are retrieved in parts by publication month |
| `build-corpus file` | Registers everything from a file of queries (see `corpus/`) |
| `add-library "folder"` | Registers the PDFs in a folder. The original files are not written to. Use `--rebuild` to register them again. Scanned PDFs cannot be registered |
| `update-integrity` | Downloads the tortured phrase list and the retraction data. The retraction data is updated every weekday, so run this before a check |
| `compact` | Merges the fingerprint index into one. Run it after registering large parts of the baseline |
| `index-citations` | Builds the list used for the citation overlap check. Run it again after adding documents |
| `index-semantic` | Builds the paraphrase search index for your PDFs. Run it again after registering the PDFs again |
| `index-semantic --pubmed` | Builds the paraphrase search index for PubMed abstracts. Set the number of documents with `--max-docs` (the default is 100,000, taken from the documents registered first) |
| `index-semantic --add-matching "regex"` | Adds PubMed abstracts whose title or journal name matches to the paraphrase search index. Build the index with `--pubmed` first. It keeps partial results in `data/simcheck/semantic_pubmed/parts/`, so it resumes after an interruption |
| `index-semantic --add-pmc` | Adds registered PMC full texts to the paraphrase search index. This is not part of the default steps (see [below](#whether-to-add-pmc-full-texts-to-the-paraphrase-search)) |
| `stats` | Shows the state of the index |
| `check "document"` | Checks a document and creates reports (see [Similarity check](#similarity-check)) |

Indexes and reports are saved in `data/simcheck/`. The index includes the text of subscription PDFs, so do not share it.

### Size and time
These values are for everything registered (about 28.4 million PubMed abstracts, about 140,000 PMC full texts, and 665 PDFs).

| Item | Value |
|---|---|
| Database (document store and citation lists) | About 59 GB |
| Merged fingerprints | About 23 GB |
| Paraphrase index (6.41 million sentences) | About 7 GB |
| Paraphrase partial results (for resuming and adding) | About 13 GB |
| Checking a manuscript of about 7,000 words | About 14 seconds without the paraphrase search and about 20 seconds with it (loading adds about 20 seconds the first time) |

### Whether to add PMC full texts to the paraphrase search
By default, the paraphrase search covers only your PDFs and the PubMed abstracts that have been indexed. It does not cover PMC body text. You can add it with `--add-pmc`, but this is not the default because the cost is high.

| Item | Current index (6.41 million sentences) | With PMC full texts (28.04 million sentences) |
|---|---:|---:|
| Heavily paraphrased PMC text detected | 0% | 94.0% |
| Lightly polished PMC text detected | 0% | 50.0% |
| Heavily paraphrased PubMed text detected | 97.9% | 95.8% |
| Candidates in text with no reuse (per 1,000 words) | 2.58 | 3.53 |
| Index size | About 7 GB | About 29 GB |
| Partial results | About 13 GB | About 55 GB |
| Peak memory during a check | About 15.5 GB | About 38.4 GB |

**`--add-pmc` replaces the default index in `data/simcheck/semantic_pubmed/`.** To keep the current index, copy the whole folder first. For details, see the [evaluation results](evaluation/detection/results.md).

## Similarity check
Use the similarity tab or the following command. PDF, DOCX, and TXT files can be checked, as well as DOC, RTF and ODT files (converted to DOCX with Word or LibreOffice). The encoding of TXT files is detected automatically.

```bash
venv\Scripts\python.exe -m simcheck check "document"
```

| Setting on screen | Command option | Default |
|---|---|---|
| Exclude references | `--include-references` to include them | Excluded |
| Exclude text in quotation marks | `--include-quotes` to include it | Excluded |
| Exclude the methods section | `--exclude-methods` | Included |
| Exclude preprints | `--exclude-preprints` | Included |
| Exclude common expressions | `--exclude-common` | Included |
| Minimum match length | `--min-words` | 8 words |
| Also check paraphrases | `--no-semantic` to turn off | On |
| Paraphrase threshold | `--semantic-threshold` | 0.85 |
| Source IDs to exclude | `--exclude-id` (can be repeated) | None |
| Allowed documents | `--allowed-doc` (can be repeated) | None |
| Web search | `--web` | Off |
| Not available | `--web-max-queries` | 300 sentences |
| Not available | `--triage` | Off |

`--triage` uses Gemini to sort the matches into stock phrases and passages to review. **The matched manuscript text and the source titles are sent to Google.** If the API key `GEMINI_API_KEY` ([API keys](#api-keys)) is not set, no sorting is done.

Allowed documents are documents that the manuscript may match, such as your own preprint, trial protocol or conference abstract (in the same formats as the checked document).
- Text that matches an allowed document for at least the minimum match length is removed from both the matches and the denominator. It is removed whichever source it matches, so you do not need to list each source ID when the same document is registered in several places.
- If the same file is registered in the corpus, that file is also excluded. Paraphrase candidates whose source sentence is the same as the allowed document are also removed.
- **If an allowed document reuses text from other publications, that text is not checked either.** Specify only your own documents.

### Web search
Sentences that did not match the sources are searched one by one as exact phrases with the Brave Search API. This looks for text reused from web pages that are not in the sources (Wikipedia, society and company sites, blogs, and so on), and it is off by default. Select web search in the similarity tab, or add `--web`.

- **The manuscript sentences are sent to Brave.** Before using it, check whether an unpublished manuscript may be sent, how Brave handles data, and what the target journal allows.
- Get the API key `BRAVE_API_KEY` from the [Brave Search API](https://brave.com/search/api/) and set it as described in [API keys](#api-keys). Each search uses your quota.
- Sentences of at least the minimum match length are searched if less than half of them is covered by matches with the sources or by excluded ranges. Sentences that are paraphrase candidates are skipped. Up to 300 sentences are searched per check by default. Searches run about once per second, so 300 sentences take about five minutes.
- The query is the first 30 words or so of the sentence (about 75 characters for Japanese). A result counts as a match only if its title and excerpt contain the same word sequence for at least the minimum match length (or the whole sentence if it is shorter). Sentences that match more than 10 results are treated as common expressions and are not listed.
- Web matches are not included in the similarity value. Only the excerpts are compared, so a sentence that is not found may still be on the web. Subscription article text rarely appears in search results. The detection rate has not been measured.
- Search results are kept in `data/simcheck/web_cache.sqlite` for 7 days, and the same sentence is not searched again. Do not share this file, because it contains manuscript sentences.

### Reports
The command saves reports in `data/simcheck/reports/` (use `--out` to change this).

- **PDF**: Contains a cover page, summary, source list, the manuscript pages, and paraphrase candidates, in that order. On the manuscript pages, matches are red, excluded ranges are gray, and paraphrase candidates have a dotted underline. Source numbers appear in the left margin. The text of the sources is not included.
- **HTML**: Shows the manuscript and the source text side by side. It includes the text of subscription PDFs, so do not share it.
- DOCX files are converted to PDF with Word to create the manuscript pages. A working copy is used, so the original file is not changed. If Word is not available, only the body text is laid out again, so pages and lines do not match the manuscript.

### Reading the results
- The similarity value is the number of matched words divided by the number of words checked. For Japanese, 2.5 characters count as 1 word.
- **Matches with documents outside the sources cannot be detected.** A low value does not prove that there is no plagiarism. The web search can only check what appears in search result excerpts.
- Commercial services use different databases and calculations, so their values cannot be compared with these. The result from a journal's check may differ.
- Stock phrases alone can produce a value when no text was reused, and the value rises with the size of the sources. With the full PubMed baseline and newly written AI text as a control, long texts (848 words on average) had a mean of 8.0% and a maximum of 15.2%, and abstract-length texts (296 words on average) had a mean of 14.2% and a maximum of 37.1%. In short texts, one matched sentence can change the value greatly, so check the matched passages rather than relying on the value alone.
- The default minimum match length is 8 words. At 7 words or fewer, matches on stock phrases increase sharply.
- The default paraphrase threshold is `0.85`. With reranking, candidates with a similarity of 0.82 or higher are narrowed down by gain. If the reranking model cannot be loaded, only the 0.85 similarity threshold is used, and the report says so.
- Lowering the threshold finds more, but in real papers it also lists many unrelated sentences from the same field.

### How detection works
1. Extracts the text from a PDF, DOCX, or TXT file (DOC, RTF and ODT files are converted to DOCX) and determines the ranges to exclude, such as references and quotations.
2. Normalizes characters (full-width and half-width forms, ligatures, line-end hyphens, invisible characters, Cyrillic and Armenian letters that look like Latin ones, Greek letters mixed into Latin words, and so on), then splits English into words and Japanese into characters.
3. Finds candidate documents with fingerprints (winnowing), then aligns tokens to locate the matches. Candidates are the top documents overall, the top documents for every 100 words of the manuscript, and documents that share a fingerprint found in 30 documents or fewer. A continuous match of 10 or more words always leaves a shared fingerprint, but on the full PubMed baseline the source may not become a candidate and the match is missed (about 5% at 10 words and about 20% at 8 words).
4. A passage that matches several sources is counted once. If you choose to exclude common expressions, expressions shared by many documents are removed from both the matches and the denominator.
5. Paraphrases are found with BAAI/bge-m3 embeddings. English sentences are then narrowed down by how much easier the sentence becomes to predict after reading the source (SCDG, Qwen2.5-1.5B). The search covers your PDFs and the PubMed abstracts that have been indexed. Above 2 million sentences, the index is divided into partitions and only 128 of them are searched (about 54 times faster than an exhaustive search at 6.41 million sentences, losing 0–3.3% of the sentences that reach a similarity of 0.82). Japanese PDFs are also searched for English sentences.

### Flags
None of these are included in the similarity value.

- **Tortured phrases**: Compares the text with the [Problematic Paper Screener](https://dbrech.irit.fr/pls/apex/f?p=9999:5) list of phrases known to be traces of paraphrasing tools. References are not checked. Two or more different phrases raise a flag, and a single phrase only produces a note.
- **Citations of retracted papers**: Compares the references with [Retraction Watch](https://gitlab.com/crossref/retraction-watch-data) data by DOI, PMID, and title. Retractions and expressions of concern are shown separately. Papers with only a correction and papers whose retraction was reversed are not shown. Title-only matches are listed separately.
- **Character manipulation**: Shows invisible characters, large numbers of soft hyphens, look-alike characters mixed into Latin words, transparent text, and so on.
- **Rendered-versus-extracted text mismatch**: Shows pages where the appearance of the PDF differs greatly from the extracted text. Up to 4 pages are rendered, read with OCR (RapidOCR) on your computer, and compared.
- **Citation overlap**: Shows a document when it shares 10 or more references with the checked document, the shared proportion is 80% or higher, and the order agrees for 70% or more. It covers documents processed by `index-citations`.

The detection rates of tortured phrases, rendered-versus-extracted text mismatch, and citation overlap have not been measured. No match does not prove that there is no problem.

### Running on GitHub Actions
See the [guide](docs/github-actions.md). **This repository is public, so do not put manuscripts in it.** To use Actions, create a separate private repository. The full manuscript is stored on GitHub.

## Revision suggestions
For each matched sentence, the tool suggests how to handle it (add a citation, quote and cite, or rewrite) and drafts a revision. It indicates a possible stock phrase only when the sentence contains no numbers and the same expression appears in many documents in the sources. This decision is not left to the LLM, because the LLM sometimes answered "no citation needed" even for sentences containing numerical results. The text is not changed automatically. A copy of the Word file, with the author's chosen actions added as comments, is saved in `data/simcheck/revisions/`.

**A rewritten sentence still needs a citation if it uses the content of the source.** Do not aim simply to lower the similarity value, and review each item. Adopted sentences count as AI-assisted editing, so declare them according to the journal's rules.

### How to use
1. Prepare an LLM provider (see the table below).
2. In the revision tab, select a DOCX file and a provider, and create the suggestions.
3. Review the suggestions one sentence at a time, edit them if needed, and choose an action.
4. Use the export button (「Wordに書き出す」) to create a copy with comments. The original file is not changed.

### Providers

| Provider | Default URL | Setup |
|---|---|---|
| LM Studio | `http://127.0.0.1:1234/v1` | Load a model and start the server |
| Strata (Qwen3.8-Flash-Next and others) | `http://127.0.0.1:8080/v1` | Start it with `strata launch`. If you set a key, also set `STRATA_API_KEY` |
| llama.cpp | `http://127.0.0.1:9931/v1` | Start `llama-server`. If the port differs, change the URL too. If you set a key, also set `LLAMA_API_KEY` |
| Gemini API (sends data externally) | Fixed | Set `GEMINI_API_KEY`. The default model is `gemini-3.8-flash` |
| Claude API (sends data externally) | Fixed | Set `ANTHROPIC_API_KEY`. The default model is `claude-opus-5-5`, and usage is billed |

- The intended local LLMs are Gemma 4, Qwen3.5, DeepSeek, Qwen3.8-Flash-Next, and similar models. Quality has been checked only for the model described in the [evaluation results](evaluation/revise/results.md).
- Only the LM Studio connection has been tried on a real server. Strata, llama.cpp, the Gemini API, and the Claude API have only been tested against simulated responses.

### When you use an external API
**The manuscript sentence, its neighbouring sentences, and the source title and excerpt are sent to an outside provider.** Before using it, check whether an unpublished manuscript may be sent, how the provider handles data, and what the target journal allows.

- An external API is used only when you select it and tick the box that agrees to sending.
- When the source is a local PDF, only its title is passed, not the excerpt, so that subscription text is not sent. Suggestions for those sentences may be weaker.
- See [API keys](#api-keys) for how to set the keys. The URLs of external APIs cannot be changed.
- A local provider cannot be pointed at a URL outside this computer.
- The Claude API request uses `fallbacks`, which retries on another model if a safety classifier declines the request.

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

**Even suggestions that pass the checks cannot be adopted as they are.** In a check with sentences from published abstracts, the meaning clearly changed in 4 of 45 suggestions that passed (for example, `One group` became `experimental group`). The checks cannot detect added facts or replaced methods or subjects, so the author must always review the content before adopting a suggestion. Only exact-match sentences are covered, not paraphrase candidates. For details, see the [evaluation results](evaluation/revise/results.md).

## Binoculars score

### Choosing models
- **Requirement**: The Observer (Base) and Performer (Instruct) must be from **the same family with the same vocabulary (tokenizer)** (for example, `Qwen/Qwen2.5-1.5B` and `Qwen/Qwen2.5-1.5B-Instruct`). Different vocabularies cause an error at load time.
- Load models by local path or Hugging Face model ID.
- A local path must be in Hugging Face format (a folder containing `config.json` and `*.safetensors`). LM Studio GGUF files cannot be loaded.
- Code in the model repository is not run (`trust_remote_code` is not used). A model that needs its own code will not load and shows an error.

### Reading the score
- The score is log PPL divided by log Cross-PPL. Lower scores tend to be closer to AI-generated text.
- No "AI-generated" or "human-written" classification or threshold is shown, because the evaluation found no threshold that gives a usable classification.
- Long documents are split every 512 tokens so that the whole text is evaluated, and the score of each segment is also shown.
- The right side of the screen uses color to show how closely each token followed the model's prediction. Green means as predicted, and red means an unexpected word. This does not indicate whether the text came from AI or a human.

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

## Evaluation results
All values apply to the conditions at the time of measurement and do not guarantee performance on real manuscripts.

| Target | Main results (full PubMed baseline) | Details |
|---|---|---|
| Similarity detection (synthetic manuscripts) | Verbatim reuse was detected at 76–81% for 8 words, about 95% for 10 words, about 98% for 12 words, and nearly 100% for 15 words or more. Editing a word every 5 words leaves no fingerprints and is rarely detected | [results.md](evaluation/detection/results.md) |
| Paraphrase candidates (synthetic manuscripts) | 82–98% of heavily paraphrased passages are listed as candidates. Translation from Japanese is at 64–86%, and this value is not stable. Text with no reuse also gets 1–3 candidates per 1,000 words | [results.md](evaluation/detection/results.md) |
| Tortured phrases and retractions | False positive checks | [results.md](evaluation/integrity/results.md) |
| Revision suggestions | Quality checks of suggestions that passed | [results.md](evaluation/revise/results.md) |
| Binoculars score | No usable threshold was found | [results.md](evaluation/ai_threshold/results.md) |

## API keys
Set the keys for external APIs as environment variables or in the API key tab (「APIキー」). Keys saved in the tab are stored per user in the Windows Credential Manager and are not shown on screen. If both are set, the environment variable is used. GitHub Actions uses only environment variables (secrets).

| Name | Used for |
|---|---|
| `BRAVE_API_KEY` | Web search |
| `GEMINI_API_KEY` | `--triage`, revision suggestions (Gemini API) |
| `ANTHROPIC_API_KEY` | Revision suggestions (Claude API) |
| `STRATA_API_KEY`, `LLAMA_API_KEY` | Revision suggestions (Strata, llama.cpp, if the server uses a key) |

## Data handling

| What you use | What is sent out | Where |
|---|---|---|
| Similarity check (without `--triage`) | Nothing | — |
| `check --triage` | Matched manuscript text and source titles | Google |
| `check --web`, web search | The first 30 words or so of sentences that did not match the sources | Brave |
| Revision suggestions (LM Studio, Strata, llama.cpp) | Nothing | — |
| Revision suggestions (Gemini API, Claude API) | The manuscript sentence and its neighbours, and the source title and excerpt (excerpts from your PDFs are not sent) | Google, Anthropic |
| GitHub Actions | The full manuscript, and the matched text if a Gemini key is set | GitHub, Google (see the [guide](docs/github-actions.md)) |
| Registering and updating sources | Search queries (no manuscript content) | NCBI and other data providers |
| Downloading models | Model IDs (no manuscript content) | Hugging Face |

- The phrase list has no stated license. Ask the provider before distributing the data in `data/integrity/` to others.
- Documents uploaded in the web UI are copied by Gradio to a temporary folder on your computer.

### Data not included
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

## Repository layout

| Location | Contents |
|---|---|
| `app.py` | Web UI (four tabs) and the Binoculars score calculation |
| `simcheck/` | The similarity check and revision suggestions. Run it with `python -m simcheck` |
| `start.bat` | Sets up the environment and starts the tool |
| `requirements.txt`, `constraints.txt` | Dependencies and their fixed versions |
| `requirements-ci.txt` | Dependencies for GitHub Actions (without PyTorch, the UI, and the paraphrase search) |
| `corpus/queries.txt` | Example queries for choosing sources |
| `evaluation/` | Evaluation methods, code, and results (`detection`, `integrity`, `revise`, `ai_threshold`) |
| `tests/` | Tests |
| `.github/workflows/`, `docs/github-actions.md`, `manuscripts/` | GitHub Actions workflows, the guide, and the folder for manuscripts |
| `binoculars/`, `datasets/`, `experiments/`, `demo/`, `assets/`, `main.py`, `setup.py` | Files from the original Binoculars |
| `data/` | Indexes, reports, and evaluation data (not included in the repository) |

The original Binoculars code is kept unchanged. This tool does not use it, and it has not been tested in this repository. `app.py` and `requirements.txt` have been replaced, so follow the instructions in the [original repository](https://github.com/ahans30/Binoculars) if you want to run the original code.

### Differences from the original Binoculars

| Item | Original Binoculars | This repository |
|---|---|---|
| AI-generated text score | Uses Falcon-7B by default and classifies text as AI-generated or human-written with a fixed threshold | Same calculation (log PPL / log Cross-PPL), implemented in `app.py`. You can choose the models, and long documents are split every 512 tokens so that the whole text is evaluated. **It gives no classification** |
| Similarity check, paraphrase candidates, flags, revision suggestions | None | Added |
| Detection evaluation | See the paper | Added (methods and results in `evaluation/`) |
| Interface | Gradio demo | Web UI with four tabs, and GitHub Actions workflows |

## Development

### Tests
```bash
venv\Scripts\python.exe -m pytest tests
```

### Updating dependencies
Each time it starts, `start.bat` installs PyTorch from the CUDA 13.2 index (`--index-url`) and everything else from PyPI, at the versions in `constraints.txt`. PyTorch is installed with `--no-deps`, because the PyTorch index does not have some of the dependency versions in `constraints.txt`. `requirements.txt` does not name a PyTorch index, because with `--extra-index-url` pip installs whichever version is higher, including a package of the same name on PyPI.

To update versions, raise them, run `pip-audit` and the tests, and then replace `constraints.txt`.

### Known unfixed vulnerability
gradio (including the latest 6.31.0) requires `pillow<13.0`, so Pillow stays on 12.x and cannot move to the fixed 13.0.0 (this includes a heap overflow in the bundled libopenjp2). Until gradio supports Pillow 13, do not load untrusted images or PDFs.

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
