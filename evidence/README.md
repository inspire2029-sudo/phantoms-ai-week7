# Evidence

This directory contains screenshots documenting the local W7D3 execution environment and results.

Recommended evidence:

1. Fedora environment, Python version, installed requirements, and Ollama model.
2. W7D3 connecting to the existing `chroma_db/w7d2_chunking` database and reusing `large_1600_250`.
3. A Top-3 retrieval example showing source pages and distances.
4. A grounded answer with its retrieved context and source pages.
5. Q9 or Q10 abstention behavior, followed by a check against the complete source PDF.
6. The final summary showing 10 RAG questions, 3 no-RAG questions, and the generated JSON/CSV files.

The JSON and CSV result files are generated locally under `W7D3/results/` and are excluded from version control by the repository ignore rules.
