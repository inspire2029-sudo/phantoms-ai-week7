# PHANTOMS AI — Week 7 / Day 1

This repository compares how chunk size and overlap affect Arabic retrieval quality in a ChromaDB-backed RAG preparation workflow.

### Source document

Use the 53-page Arabic/English technical report:

**SGDS_SWARM_Engineering_Report_Civil_Edition.pdf**

Place the PDF at:

~~~text
data/source_document.pdf
~~~

The script extracts each PDF page separately. Each chunk stores its source filename and 1-based PDF page number.

### Experiment design

| Configuration | Chunk size | Overlap | Description |
|---|---:|---:|---|
| small_180_30 | 180 characters | 30 | Very small |
| medium_700_120 | 700 characters | 120 | Medium |
| large_1600_250 | 1600 characters | 250 | Very large |

- Embedding model: \`intfloat/multilingual-e5-small\` (FastEmbed).
- Vector store: ChromaDB with cosine distance.
- Retrieval: Top 3 chunks for each of 10 Arabic questions.
- Questions Q9 and Q10 are deliberately unanswerable from the source.
- Q1–Q6 require evidence from multiple pages, so a complete hit requires every expected evidence page to appear in the Top 3.

### Run on Fedora

~~~bash
cd ~/Projects/phantoms-ai-week7
mkdir -p data results
python -m pip install -r requirements.txt
python W7D1_Vector_DB.py
~~~

The script writes \`results/comparison_results.csv\` and updates this README with the measured 30-row comparison, summary, best configuration, and worst failure.

---

## W7D2 — Chunking and Retrieval Quality Benchmark

The complete Day 2 deliverable is consolidated into one Markdown file:

- [W7D2_Vector_DB_Chunking.md](W7D2_Vector_DB_Chunking.md) — Egyptian Arabic first-person explanation, source document, 10 questions, three chunking configurations, metadata design, measured comparison, analysis, and the complete Python implementation in a dedicated code section.

The standalone \`W7D2_Chunking_Experiment.py\` file was removed. The Python implementation is embedded in the Markdown deliverable.

**Selected W7D2 configuration:** \`large_1600_250\` — 1600 characters with 250 overlap. It achieved 55.1% evidence coverage and completed 4 of 8 answerable questions in the recorded benchmark. Medium was close at 53.8%, so this selection is specific to this document and question set.

---

## W7D3 — RAG Pipeline and Grounded Generation

The complete Day 3 deliverable is consolidated into one Markdown file:

- [W7D3_RAG_Pipeline.md](W7D3_RAG_Pipeline.md) — Egyptian Arabic first-person explanation, full RAG workflow, Top-k reasoning, grounded System Prompt, the same 10 questions, failure analysis, and the complete Python implementation in a dedicated code section.

### Implementation

- Reuses the W7D2 persistent database at \`chroma_db/w7d2_chunking\`.
- Reuses the existing \`large_1600_250\` collection without rebuilding it.
- Uses the same multilingual embedding model and Top 3 retrieval.
- Sends retrieved context and source metadata to an OpenAI-compatible Chat Completions endpoint.
- Runs all 10 questions with RAG and Q1, Q9, Q10 without RAG.
- Saves local execution output to \`results/W7D3_rag_results.json\` and \`results/W7D3_rag_results.csv\`.

### Run W7D3

Extract the Python code block from the W7D3 Markdown deliverable into a temporary file named \`W7D3_RAG_Pipeline.py\`, then configure the LLM environment:

~~~bash
export LLM_API_KEY="YOUR_API_KEY"
export LLM_BASE_URL="https://api.openai.com/v1"
export LLM_MODEL="gpt-4o-mini"
python W7D3_RAG_Pipeline.py
~~~

The API key must remain in an environment variable and must never be committed.

**Execution status:** The implementation and documentation are committed. Actual LLM outputs and their correctness labels must be produced by running against the local W7D2 Chroma database and a valid API key. No model responses or execution measurements are fabricated in this README.
