# PHANTOMS AI — Week 7

This repository contains the Week 7 Vector Database, Chunking, and RAG exercises.

## Source document

Use the same 53-page technical report from W7D1 and W7D2:

**SGDS_SWARM_Engineering_Report_Civil_Edition(4) (1).pdf**

Local path:

~~~text
data/source_document.pdf
~~~

The PDF and local ChromaDB data are not committed to this repository.

---

## W7D1 — Vector Database

The first exercise builds the ChromaDB vector database and prepares the source document for semantic retrieval.

- [W7D1_Vector_DB.py](W7D1_Vector_DB.py)
- [Requirements](requirements.txt)

---

## W7D2 — Chunking and Retrieval Quality Benchmark

The complete Day 2 deliverable is consolidated into one Markdown file:

- [W7D2_Vector_DB_Chunking.md](W7D2_Vector_DB_Chunking.md) — Egyptian Arabic first-person explanation, the source document, the same 10 questions, three chunking configurations, metadata design, measured comparison, analysis, and the complete Python implementation in a dedicated code section.

### Recorded benchmark

| Configuration | Chunks | Evidence coverage | Completed answerable questions |
|---|---:|---:|---:|
| Small 180/30 | 429 | 28.2% | 1 / 8 |
| Medium 700/120 | 120 | 53.8% | 3 / 8 |
| Large 1600/250 | 62 | 55.1% | 4 / 8 |

**Selected W7D2 configuration:** \`large_1600_250\` — 1600 characters with 250 overlap. It achieved the highest overall evidence coverage in the recorded benchmark. Medium was close at 53.8%, so this selection is specific to this document and question set.

---

## W7D3 — RAG Pipeline and Grounded Generation

The complete Day 3 explanation and implementation are consolidated into one Markdown file:

- [W7D3_RAG_Pipeline.md](W7D3_RAG_Pipeline.md) — Egyptian Arabic first-person explanation, full RAG workflow, Top-k reasoning, grounded System Prompt, the same 10 questions, RAG versus no-RAG comparison, failure analysis, and the complete Python implementation in a dedicated code section.

### Pipeline configuration

| Item | Value |
|---|---|
| Persistent database | \`chroma_db/w7d2_chunking\` |
| Collection | \`large_1600_250\` |
| Expected chunks | 62 |
| Embedding model | \`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2\` |
| Retrieval | Top 3 |
| RAG questions | 10 |
| No-RAG questions | Q1, Q9, Q10 |
| Source metadata | \`source_file\`, \`page_number\` |

The W7D3 implementation opens the existing W7D2 collection. It does not recreate the collection or change the chunking configuration.

### Table of the 10 questions and evaluation

The exact questions and reference answers are documented in W7D2 and repeated in the W7D3 deliverable. The following table is the W7D3 evaluation register. Model answers and correctness labels are intentionally left pending until an actual run is reviewed against the source.

| ID | Question | Reference evidence | RAG answer / assessment |
|---|---|---|---|
| Q1 | ما الدور الذي تمثله كل طائرة داخل منظومة SGDS SWARM؟ | صفحة 3 | Pending execution and review |
| Q2 | ما الطبقات الوظيفية العشر للمنظومة وما الطبقتان العرضيتان اللتان تغطيانها؟ | صفحتا 8 و 9 | Pending execution and review |
| Q3 | لماذا لا يعتمد التقرير على عتبة بطارية ثابتة وما المدخلات المستخدمة لتقدير زمن المهمة الآمن؟ | صفحتا 13 و 14 | Pending execution and review |
| Q4 | ما المقاييس التشغيلية التي يوصي التقرير بتسجيلها بجانب mAP ولماذا؟ | صفحتا 24 و 25 | Pending execution and review |
| Q5 | كيف يتعامل النظام مع تراجع أداء إصدار نموذج جديد وكيف يراقب تغير البيانات؟ | صفحتا 26 و 27 | Pending execution and review |
| Q6 | ما الإجراءات المطلوبة لإدارة مفاتيح وحدات الأسطول وتأمين تحديثات البرامج؟ | صفحة 32 | Pending execution and review |
| Q7 | ما المبدأ الذي يجب أن يحكم تصرف الطائرة عند فقدان اتصال الشبكة؟ | صفحة 44 | Pending execution and review |
| Q8 | ما التسليمات المطلوبة من فريق الذكاء الاصطناعي بخلاف ملف model.pt؟ | صفحة 49 | Pending execution and review |
| Q9 | ما قيمة mAP التي حققها النموذج فعليًا في اختبار ميداني موثق على جهاز Jetson المستهدف؟ | غير موجودة في المستند | Pending abstention check |
| Q10 | كم رحلة اختبار ميداني مكتملة نُفذت فعليًا وما متوسط زمن الطيران المقاس بالدقائق؟ | غير موجودة في المستند | Pending abstention check |

### Comparison with and without RAG

The no-RAG run uses Q1, Q9, and Q10. The RAG run uses the same questions plus the remaining seven questions.

| ID | No-RAG | With RAG | What is being tested |
|---|---|---|---|
| Q1 | Pending execution | Pending execution | Whether source context improves a factual answer |
| Q9 | Pending execution | Pending execution | Whether the model invents a field mAP value or abstains |
| Q10 | Pending execution | Pending execution | Whether the model invents flight counts or flight duration |

The code saves the actual responses for both modes. These cells remain pending because the local ChromaDB database and a valid LLM API key are required. No generated answers or correctness results are fabricated.

### Failure analysis — Q4

Q4 is the selected retrieval-failure case for investigation.

In the recorded W7D2 Large benchmark, Q4 retrieved **0 of 7 expected evidence anchors** in the Top 3. The evidence is distributed across PDF pages 24 and 25 and includes False Alerts/hour, Miss Rate, False Alarms per Operational Hour, Inference Latency, FPS, Watt/Inference, and Thermal Throttling Rate.

The initial diagnosis is a **Retrieval failure**: the required evidence did not appear in the selected Top 3. If the LLM subsequently produces an incomplete answer, that alone does not prove a Generation failure because the relevant evidence may never have reached the prompt.

Planned changes to test, without treating them as proven improvements:

1. Query expansion with the exact metric names.
2. Compare Top 3 against Top 5 and Top 7.
3. Inspect the chunks around pages 24 and 25 for split lists or missing context.
4. Evaluate hybrid semantic and keyword retrieval.
5. Compare evidence coverage and answer correctness against the same ground truth.

This diagnosis is based on the measured W7D2 retrieval benchmark. The W7D3 run must be executed before claiming that the same failure occurred in the end-to-end pipeline.

### Run W7D3

Extract the Python code block from the W7D3 Markdown deliverable into a temporary file named \`W7D3_RAG_Pipeline.py\`, then configure the LLM environment:

~~~bash
cd ~/Projects/phantoms-ai-week7
export LLM_API_KEY="YOUR_API_KEY"
export LLM_BASE_URL="https://api.openai.com/v1"
export LLM_MODEL="gpt-4o-mini"
python W7D3_RAG_Pipeline.py
~~~

The API key must remain in an environment variable and must never be committed.

### Local execution outputs

The script writes:

- \`results/W7D3_rag_results.json\` — generated answers, context, retrieved chunks, and source metadata.
- \`results/W7D3_rag_results.csv\` — a review-friendly table for all RAG and no-RAG outputs.

The final correctness labels must be assigned after comparing each generated answer with the source document. The script deliberately marks them as pending manual review rather than claiming unverified answers are correct.

**Execution status:** W7D3 code and documentation are committed. Actual LLM outputs and correctness labels still require execution against the local W7D2 Chroma database and a valid API key.
