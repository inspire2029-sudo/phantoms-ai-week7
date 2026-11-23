# PHANTOMS AI — Week 7

This repository contains the Week 7 vector database, chunking benchmark, and grounded RAG pipeline exercises.

## Repository layout

```text
.
├── W7D1_Vector_DB.py
├── W7D2_Vector_DB_Chunking.md
├── W7D3/
│   ├── README.md
│   └── W7D3_RAG_Pipeline.py
├── data/
│   └── README.md
├── evidence/
│   └── README.md
├── requirements.txt
└── .gitignore
```

## Source document

Use the same 53-page source report for all three days:

`SGDS_SWARM_Engineering_Report_Civil_Edition(4) (1).pdf`

Place it locally at `data/source_document.pdf`. The source PDF and generated ChromaDB data are intentionally not committed.

## Week 7 deliverables

### Day 1 — Vector Database

- [W7D1_Vector_DB.py](W7D1_Vector_DB.py)

### Day 2 — Chunking and retrieval benchmark

- [W7D2_Vector_DB_Chunking.md](W7D2_Vector_DB_Chunking.md)

The documented selected configuration is Large: 1600 characters with 250-character overlap, stored in the `large_1600_250` collection under `chroma_db/w7d2_chunking`.

### Day 3 — RAG Pipeline

- [W7D3 documentation and runbook](W7D3/README.md)
- [W7D3 executable pipeline](W7D3/W7D3_RAG_Pipeline.py)

W7D3 reuses the existing W7D2 Chroma collection. It does not silently rebuild a replacement database. The pipeline runs ten questions with RAG and Q1, Q9, and Q10 without RAG. It checkpoints JSON and CSV results after every question so interrupted runs can resume.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For Fedora, follow the detailed Ollama and execution instructions in [W7D3/README.md](W7D3/README.md).

## Evidence and generated files

- Keep the original PDF at `data/source_document.pdf` locally.
- ChromaDB files and run outputs are generated locally and ignored by Git.
- Save genuine terminal screenshots in `evidence/` only after the corresponding commands succeed.

See each day’s document for methodology, configuration, evaluation criteria, and known limitations.
