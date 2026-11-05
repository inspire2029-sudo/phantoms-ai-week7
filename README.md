# PHANTOMS AI — Week 7

## W7D1 — Vector Database with Chroma

This task builds on the Week 6 work with Embeddings, Cosine Similarity, Euclidean Distance, and Mini Semantic Search.

## Files

- W7D1_Vector_DB.py — creates a persistent Chroma database, stores 12 documents, runs five semantic queries, compares filtered and unfiltered results, and updates the measured results in notes.
- notes.md — documents the experiment, the SQL vs Vector Database comparison, the distance metric, the out-of-scope query, and the measured threshold analysis.

## Setup

python -m pip install chromadb

## Run

python W7D1_Vector_DB.py

Run the script twice to verify that the collection still contains 12 documents. The script uses stable IDs and upsert so rerunning it updates the same records instead of inserting duplicates.

## Important experiment notes

- The collection uses Chroma's default distance configuration.
- The default Single Node Chroma distance metric is squared L2 distance.
- Lower distance means a closer result.
- A nearest-neighbor result is not automatically a relevant answer. Q5 tests this limitation with an out-of-scope question.
- The threshold is calculated from the actual distances produced by the local run. It is not hardcoded.

## Learning direction

Build it. Break it. Understand it. Document it.
