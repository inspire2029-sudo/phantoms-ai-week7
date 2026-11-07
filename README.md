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

Today's experiment is kept in a separate script so the existing Day 1 work remains untouched.

### Script

- `W7D2_Chunking_Experiment.py` — builds three Chroma collections, retrieves Top 3 for the same 10 Arabic questions, checks manually defined evidence anchors, and validates source metadata.

### Fixed experiment variables

| Configuration | Chunk size | Overlap |
|---|---:|---:|
| small_180_30 | 180 characters | 30 characters |
| medium_700_120 | 700 characters | 120 characters |
| large_1600_250 | 1600 characters | 250 characters |

The source PDF must be available locally at `data/source_document.pdf`. It is not stored in this repository.

### Run

```bash
cd ~/Projects/phantoms-ai-week7
python W7D2_Chunking_Experiment.py
```

### Generated deliverables

- `results/W7D2_Chunking_Experiment.md` — questions, reference answers, the 10 × 3 comparison table, measured winner, worst failure, concepts, and metadata verification.
- `results/W7D2_chunking_comparison.csv` — all 30 query/configuration rows with Top 3 pages, distances, chunk IDs, and source metadata.

**Status:** implementation added; measured results remain pending the local run against the source PDF. Never fill the comparison with estimated or invented retrieval results.
