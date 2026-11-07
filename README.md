# PHANTOMS AI — Week 7 / Day 1
## Arabic Vector Database: Chunk Size & Overlap Benchmark

This repository compares how chunk size and overlap affect Arabic retrieval quality in a ChromaDB-backed RAG preparation workflow.

### Source document

Use the 53-page Arabic/English technical report:

**SGDS_SWARM_Engineering_Report_Civil_Edition.pdf**

Place the PDF at:

```text
data/source_document.pdf
```

The script extracts each PDF page separately. Each chunk stores its source filename and 1-based PDF page number.

### Experiment design

| Configuration | Chunk size | Overlap | Description |
|---|---:|---:|---|
| small_180_30 | 180 characters | 30 | Very small |
| medium_700_120 | 700 characters | 120 | Medium |
| large_1600_250 | 1600 characters | 250 | Very large |

- Embedding model: `intfloat/multilingual-e5-small` (FastEmbed).
- Vector store: ChromaDB with cosine distance.
- Retrieval: Top 3 chunks for each of 10 Arabic questions.
- Questions Q9 and Q10 are deliberately unanswerable from the source.
- Q1–Q6 require evidence from multiple pages, so a complete hit requires every expected evidence page to appear in the Top 3.

### Run on Fedora

```bash
cd ~/Projects/phantoms-ai-week7
mkdir -p data results
# Copy the downloaded PDF to data/source_document.pdf
python -m pip install -r requirements.txt
python W7D1_Vector_DB.py
```

The script writes `results/comparison_results.csv` and updates this README with the measured 30-row comparison, summary, best configuration, and worst failure.

### Evidence to capture

1. Environment and dependency versions.
2. PDF page-count validation.
3. Chunk counts for all three configurations.
4. Top-3 retrieval output for all 30 tests.
5. Final CSV row count and Metadata verification (30/30 expected).

**Important:** Do not invent or manually fill retrieval results. The final analysis must be based on the actual terminal run.

---

## W7D2 — Chunking and Retrieval Quality Benchmark

The complete Day 2 deliverable is consolidated into one Markdown file:

- [W7D2_Vector_DB_Chunking.md](W7D2_Vector_DB_Chunking.md) — Egyptian Arabic first-person explanation, source document, 10 questions, three chunking configurations, metadata design, result-table template, analysis criteria, and the complete Python implementation in a dedicated code section.

The standalone `W7D2_Chunking_Experiment.py` file was removed. To run the experiment, extract the Python code block from the Markdown deliverable into a temporary local file named `W7D2_Chunking_Experiment.py`, place the source PDF at `data/source_document.pdf`, install `requirements.txt`, and run the script.

**Status:** W7D2 documentation is finalized. The report records the completed 30-run comparison (10 questions × 3 configurations), measured evidence-coverage results, and the interpretation/limitations. Q9/Q10 remain separate source-absence checks and are not counted as retrieval successes or failures. The Python implementation is embedded in the Markdown deliverable; the source PDF, CSV, and local ChromaDB files remain local and are not committed.
