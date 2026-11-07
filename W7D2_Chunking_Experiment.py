from __future__ import annotations
import csv, re, statistics, unicodedata
from pathlib import Path
import chromadb
from fastembed import TextEmbedding
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
PDF = ROOT / "data" / "source_document.pdf"
DB = ROOT / "chroma_db" / "w7d2_chunking"
OUT = ROOT / "results"
CSV_FILE = OUT / "W7D2_chunking_comparison.csv"
REPORT = OUT / "W7D2_Chunking_Experiment.md"
SOURCE = "SGDS_SWARM_Engineering_Report_Civil_Edition.pdf"
MODEL = "intfloat/multilingual-e5-small"
TOP_K = 3

CONFIGS = [
    {"name": "small_180_30", "size": 180, "overlap": 30},
    {"name": "medium_700_120", "size": 700, "overlap": 120},
    {"name": "large_1600_250", "size": 1600, "overlap": 250},
]

QUESTIONS = [
    {"id": "Q1", "question": "ما الدور الذي تمثله كل طائرة داخل منظومة SGDS SWARM؟",
     "answer": "كل طائرة تمثل عقدة استشعار ومعالجة تنفذ الطيران وتستخلص المعلومات وتقيّم حالتها التشغيلية وتتبادل بيانات مختصرة وتتخذ قرارات محلية ضمن السياسات المحددة.",
     "evidence": [{"page": 3, "anchors": ["عقدة استشعار ومعالجة", "استخلاص المعلومات", "قرارات محلية"]}]},
    {"id": "Q2", "question": "ما الطبقات الوظيفية العشر للمنظومة وما الطبقتان العرضيتان اللتان تغطيانها؟",
     "answer": "الطبقات الوظيفية هي Airframe / Power / Mechanics وSensors / Payload وFlight Control وNavigation / Localization وCommunications وData Management وAI / Computer Vision وEvent & Decision Engine وFleet Management وUser Interface / Command Center. والطبقتان العرضيتان هما Cybersecurity وSafety / Reliability.",
     "evidence": [{"page": 8, "anchors": ["عشر طبقات وظيفية", "User Interface / Command Center", "Airframe / Power / Mechanics"]},
                  {"page": 9, "anchors": ["Cybersecurity", "Safety / Reliability"]}]},
    {"id": "Q3", "question": "لماذا لا يعتمد التقرير على عتبة بطارية ثابتة وما المدخلات المستخدمة لتقدير زمن المهمة الآمن؟",
     "answer": "لأن نسبة البطارية لا تمثل الاحتياطي نفسه في كل الظروف إذ يتغير الاستهلاك حسب الحمولة والطقس والمسافة وحالة البطارية. ويعتمد التقدير على الطاقة المتبقية وطاقة العودة واحتياطي السلامة وتأثير الطقس وصحة البطارية واستهلاك الحمولة.",
     "evidence": [{"page": 13, "anchors": ["لا تمثل مقدار الاحتياطي نفسه", "يتغير الاستهلاك المتوقع", "Energy-Aware Mission Management"]},
                  {"page": 14, "anchors": ["Remaining Energy", "Return Energy", "Safety Reserve", "Payload Consumption"]}]},
    {"id": "Q4", "question": "ما المقاييس التشغيلية التي يوصي التقرير بتسجيلها بجانب mAP ولماذا؟",
     "answer": "يوصي بتسجيل False Alarms per Operational Hour إضافة إلى Inference Latency وFPS وWatt/Inference وThermal Throttling Rate لأن ارتفاع mAP وحده لا يكشف كثرة الإنذارات الكاذبة أو تكلفة وسرعة التشغيل على العتاد.",
     "evidence": [{"page": 25, "anchors": ["False Alarms per Operational Hour", "Inference Latency", "Thermal Throttling Rate"]}]},
    {"id": "Q5", "question": "كيف يتعامل النظام مع تراجع أداء إصدار نموذج جديد وكيف يراقب تغير البيانات؟",
     "answer": "يرصد Health Monitor تراجع الأداء ثم ينفذ Rollback إلى الإصدار السابق. ويراقب Data Drift من خلال Average confidence وUnknown detections وFalse alarms وScene distribution وImage quality ثم يطلب Model Review Required عند اكتشاف الانحراف.",
     "evidence": [{"page": 26, "anchors": ["Health monitor detects degradation", "Rollback", "Model v1.4 restored"]},
                  {"page": 27, "anchors": ["Average confidence", "Unknown detections", "Scene distribution", "Model Review Required"]}]},
    {"id": "Q6", "question": "ما الإجراءات المطلوبة لإدارة مفاتيح وحدات الأسطول وتأمين تحديثات البرامج؟",
     "answer": "يجب تحديد مسؤولية إنشاء المفاتيح وتوزيعها وتخزينها وصلاحيات الوصول وتدويرها وإلغائها وإجراء فقدان الجهاز. والتحديث الآمن يمر بحزمة موقعة ثم التحقق من التوقيع والتوافق والتثبيت وفحص الصحة مع الرجوع للإصدار السابق عند الفشل.",
     "evidence": [{"page": 32, "anchors": ["إنشاء المفاتيح وتوزيعها", "تدوير المفاتيح وإلغائها", "Signed Package", "Signature Verification", "Rollback"]}]},
    {"id": "Q7", "question": "ما المبدأ الذي يجب أن يحكم تصرف الطائرة عند فقدان اتصال الشبكة؟",
     "answer": "فقدان الشبكة لا يعني فقدان الطائرة ويجب أن تحتفظ كل طائرة بمنطق سلامة محلي وألا تنتظر أوامر الخادم عند انقطاع الاتصال.",
     "evidence": [{"page": 44, "anchors": ["Loss of Network", "Loss of Aircraft", "Local Safety Logic", "فقدان الاتصال"]}]},
    {"id": "Q8", "question": "ما التسليمات المطلوبة من فريق الذكاء الاصطناعي بخلاف ملف model.pt؟",
     "answer": "تشمل Dataset specification وDataset version وLabel policy وTraining pipeline وEvaluation report وModel artifact وTensorRT artifact وBenchmark وInference service وEvent API وModel card وFailure cases وDrift monitor وRollback version.",
     "evidence": [{"page": 49, "anchors": ["Dataset specification", "Dataset version", "Label policy", "Training pipeline", "Evaluation report"]},
                  {"page": 49, "anchors": ["TensorRT artifact", "Benchmark", "Inference service", "Event API", "Model card", "Failure cases", "Drift monitor", "Rollback version"]}]},
    {"id": "Q9", "question": "ما قيمة mAP التي حققها النموذج فعليًا في اختبار ميداني موثق على جهاز Jetson المستهدف؟",
     "answer": "غير موجودة في المستند. يذكر التقرير مقاييس يجب قياسها ولا يقدم قيمة mAP موثقة من اختبار ميداني على الجهاز المستهدف.", "evidence": []},
    {"id": "Q10", "question": "كم رحلة اختبار ميداني مكتملة نُفذت فعليًا وما متوسط زمن الطيران المقاس بالدقائق؟",
     "answer": "غير موجود في المستند. لا يقدم التقرير عدد رحلات مكتملة أو متوسط زمن طيران مقاسًا بالدقائق.", "evidence": []},
]

def norm(value):
    value = unicodedata.normalize("NFKC", value or "")
    value = "".join(c for c in value if unicodedata.category(c) != "Mn").replace("ـ", "")
    value = re.sub(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff]", "", value)
    return re.sub(r"\s+", " ", value).casefold().strip()

def load_pages():
    if not PDF.exists():
        raise FileNotFoundError(f"ملف المصدر غير موجود: {PDF}\nضعي التقرير داخل data باسم source_document.pdf")
    reader = PdfReader(str(PDF))
    if len(reader.pages) < 5:
        raise ValueError("المستند أقل من خمس صفحات")
    pages = []
    for number, page in enumerate(reader.pages, 1):
        text = unicodedata.normalize("NFC", page.extract_text() or "")
        text = re.sub(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff]", "", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if text:
            pages.append({"page": number, "text": text})
    if len(pages) < 5:
        raise ValueError(f"النص قابل للاستخراج من {len(pages)} صفحات فقط")
    return pages, len(reader.pages)

def split_chunks(pages, cfg):
    size, overlap = cfg["size"], cfg["overlap"]
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Chunk Size وOverlap غير صالحين")
    chunks, step = [], size - overlap
    for page in pages:
        start = index = 0
        while start < len(page["text"]):
            part = page["text"][start:start + size].strip()
            if part:
                chunks.append({
                    "id": f"{cfg['name']}_p{page['page']:03d}_c{index:04d}",
                    "text": part,
                    "meta": {"source_file": SOURCE, "page_number": int(page["page"]),
                             "chunk_index": int(index), "chunk_size_chars": int(size),
                             "overlap_chars": int(overlap), "chunking_method": "fixed_character_per_page"}
                })
                index += 1
            if start + size >= len(page["text"]):
                break
            start += step
    return chunks

def index_chunks(collection, model, chunks):
    for start in range(0, len(chunks), 64):
        batch = chunks[start:start + 64]
        vectors = [v.tolist() for v in model.embed(["passage: " + x["text"] for x in batch])]
        collection.upsert(ids=[x["id"] for x in batch],
                          documents=[x["text"] for x in batch],
                          metadatas=[x["meta"] for x in batch], embeddings=vectors)

def retrieve(collection, model, q):
    vector = next(model.embed(["query: " + q["question"]])).tolist()
    result = collection.query(query_embeddings=[vector], n_results=min(TOP_K, collection.count()),
                              include=["documents", "metadatas", "distances"])
    hits = []
    for rank, (cid, doc, meta, distance) in enumerate(
        zip(result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]), 1):
        hits.append({"rank": rank, "id": cid, "doc": doc, "meta": meta, "distance": float(distance)})
    metadata_ok = all(h["meta"].get("source_file") == SOURCE
                      and isinstance(h["meta"].get("page_number"), int)
                      and h["meta"]["page_number"] > 0 for h in hits)
    evidence = []
    for group in q["evidence"]:
        match = None
        for hit in hits:
            if hit["meta"].get("page_number") != group["page"]:
                continue
            doc = norm(hit["doc"])
            anchor = next((a for a in group["anchors"] if norm(a) in doc), None)
            if anchor:
                match = {"page": group["page"], "rank": hit["rank"], "found": True}
                break
        evidence.append(match or {"page": group["page"], "rank": None, "found": False})
    if not q["evidence"]:
        status, coverage = "NOT_IN_SOURCE", None
    else:
        found = sum(x["found"] for x in evidence)
        coverage = found / len(evidence)
        status = "YES" if found == len(evidence) else ("PARTIAL" if found else "NO")
    return {"hits": hits, "metadata_ok": metadata_ok, "evidence": evidence,
            "status": status, "coverage": coverage}

def main():
    OUT.mkdir(exist_ok=True)
    DB.mkdir(parents=True, exist_ok=True)
    pages, page_count = load_pages()
    print(f"[SOURCE] PDF pages={page_count} | text pages={len(pages)}")
    print(f"[MODEL] Loading {MODEL}; first run may download model files")
    model = TextEmbedding(model_name=MODEL)
    client = chromadb.PersistentClient(path=str(DB))
    rows, stats = [], {}

    for cfg in CONFIGS:
        chunks = split_chunks(pages, cfg)
        try:
            client.delete_collection(name=cfg["name"])
        except Exception:
            pass
        col = client.create_collection(name=cfg["name"], metadata={"hnsw:space": "cosine"})
        index_chunks(col, model, chunks)
        print(f"\n[CONFIG] {cfg['name']} size={cfg['size']} overlap={cfg['overlap']} chunks={col.count()}")
        complete = partial = failed = 0
        coverages, first_ranks = [], []
        for q in QUESTIONS:
            r = retrieve(col, model, q)
            if q["evidence"]:
                coverages.append(r["coverage"])
                found_ranks = [x["rank"] for x in r["evidence"] if x["found"]]
                complete += r["status"] == "YES"
                partial += r["status"] == "PARTIAL"
                failed += r["status"] == "NO"
                if found_ranks:
                    first_ranks.append(min(found_ranks))
            top_pages = " > ".join(f"p.{h['meta'].get('page_number', '?')}" for h in r["hits"])
            order = "; ".join(f"p.{x['page']}→#{x['rank']}" if x["found"] else f"p.{x['page']}→missing"
                              for x in r["evidence"]) or "غير موجود في المصدر"
            rows.append({"qid": q["id"], "question": q["question"], "config": cfg["name"],
                         "size": cfg["size"], "overlap": cfg["overlap"],
                         "expected": ", ".join(str(x["page"]) for x in q["evidence"]) or "ABSENT",
                         "status": r["status"], "order": order, "top_pages": top_pages,
                         "metadata": r["metadata_ok"], "hits": r["hits"]})
            print(f"  {q['id']} [{r['status']}] expected={rows[-1]['expected']} | top3={top_pages} | metadata={'OK' if r['metadata_ok'] else 'ERROR'}")
        stats[cfg["name"]] = {"complete": int(complete), "partial": int(partial), "failed": int(failed),
                              "coverage": statistics.mean(coverages), "rank": statistics.mean(first_ranks) if first_ranks else 4,
                              "chunks": col.count()}
        s = stats[cfg["name"]]
        print(f"[SUMMARY] complete={s['complete']}/8 partial={s['partial']}/8 failed={s['failed']}/8 coverage={s['coverage']:.1%}")

    fields = ["question_id", "question", "configuration", "chunk_size_chars", "overlap_chars",
              "expected_pages", "answer_in_top3", "evidence_rank_order", "top3_pages", "metadata_verified",
              "top1_page", "top1_distance", "top1_source", "top1_chunk_id", "top1_preview",
              "top2_page", "top2_distance", "top2_source", "top2_chunk_id", "top2_preview",
              "top3_page", "top3_distance", "top3_source", "top3_chunk_id", "top3_preview"]
    with CSV_FILE.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {"question_id": row["qid"], "question": row["question"], "configuration": row["config"],
                   "chunk_size_chars": row["size"], "overlap_chars": row["overlap"],
                   "expected_pages": row["expected"], "answer_in_top3": row["status"],
                   "evidence_rank_order": row["order"], "top3_pages": row["top_pages"],
                   "metadata_verified": row["metadata"]}
            for i in range(1, 4):
                h = row["hits"][i - 1] if len(row["hits"]) >= i else None
                out[f"top{i}_page"] = h["meta"].get("page_number", "") if h else ""
                out[f"top{i}_distance"] = f"{h['distance']:.6f}" if h else ""
                out[f"top{i}_source"] = h["meta"].get("source_file", "") if h else ""
                out[f"top{i}_chunk_id"] = h["id"] if h else ""
                out[f"top{i}_preview"] = re.sub(r"\s+", " ", h["doc"])[:220] if h else ""
            writer.writerow(out)

    failures = [r for r in rows if r["status"] in {"NO", "PARTIAL"}]
    if failures:
        worst = min(failures, key=lambda r: (r["status"] == "PARTIAL", -r["order"].count("missing")))
        worst_text = (f"أسوأ حالة هي {worst['qid']} مع {worst['config']}؛ أعلى 3 نتائج أعادت {worst['top_pages']} "
                      f"ولم تغطِّ كل الأدلة المطلوبة ({worst['order']}). راجعي مقتطفات CSV لتحديد هل السبب حدّ القطعة "
                      "أو تشتت الدليل أو منافسة مقاطع أخرى.")
    else:
        worst_text = "لم يظهر فشل كامل. راجعي حالات PARTIAL إن وجدت، وإلا قارني أقل تغطية وأبطأ ترتيب لأول دليل."

    winner_name, winner = max(stats.items(), key=lambda x: (x[1]["coverage"], x[1]["complete"], -x[1]["rank"]))
    meta_total = sum(len(r["hits"]) for r in rows)
    meta_valid = sum(1 for r in rows for h in r["hits"]
                     if h["meta"].get("source_file") == SOURCE
                     and isinstance(h["meta"].get("page_number"), int) and h["meta"]["page_number"] > 0)

    report = [
        "# W7D2 — Chunking وتأثيره على جودة Retrieval في RAG", "",
        "## هدف التجربة", "",
        "قياس أثر Chunk Size وOverlap على استرجاع الأدلة من مستند عربي مع تثبيت المصدر والأسئلة ونموذج Embedding وعمق الاسترجاع.", "",
        "## المستند", "", f"- الملف: {SOURCE}", f"- صفحات PDF: {page_count}",
        f"- صفحات بها نص قابل للاستخراج: {len(pages)}", "- أرقام الصفحات هي أرقام PDF الفعلية وتبدأ من 1.", "",
        "## الإعدادات", "", "| الإعداد | الحجم بالحروف | التداخل بالحروف | الوصف |",
        "|---|---:|---:|---|", "| small_180_30 | 180 | 30 | صغير جدًا |",
        "| medium_700_120 | 700 | 120 | متوسط |", "| large_1600_250 | 1600 | 250 | كبير جدًا |", "",
        f"- Embedding: {MODEL}", "- Vector DB: ChromaDB", "- Distance: cosine", "- Retrieval: Top 3",
        "- القياس بالحروف وليس Tokens.", "", "## الأسئلة والإجابات المرجعية", ""
    ]
    for q in QUESTIONS:
        expected = ", ".join(str(x["page"]) for x in q["evidence"]) or "غير موجودة"
        report += [f"### {q['id']} — {q['question']}", f"- صفحات الدليل: {expected}", f"- الإجابة: {q['answer']}", ""]
    report += ["## جدول المقارنة — 10 أسئلة × 3 إعدادات", "",
               "| السؤال | الإعداد | صفحات الدليل | أعلى 3 بالترتيب | الدليل موجود؟ | ترتيب الأدلة | Metadata |",
               "|---|---|---|---|---|---|---|"]
    for r in rows:
        report.append(f"| {r['qid']} | {r['config']} | {r['expected']} | {r['top_pages']} | {r['status']} | {r['order']} | {'OK' if r['metadata'] else 'ERROR'} |")
    report += ["", "## ملخص القياس", "", "| الإعداد | كامل من 8 | جزئي | فشل | تغطية الأدلة | متوسط ترتيب أول دليل | عدد القطع |",
               "|---|---:|---:|---:|---:|---:|---:|"]
    for cfg in CONFIGS:
        s = stats[cfg["name"]]
        report.append(f"| {cfg['name']} | {s['complete']}/8 | {s['partial']}/8 | {s['failed']}/8 | {s['coverage']:.1%} | {s['rank']:.2f} | {s['chunks']} |")
    report += ["", "## التحليل", "",
        f"الإعداد الأفضل هو {winner_name} لأنه حقق تغطية أدلة {winner['coverage']:.1%} وأكمل {winner['complete']} إجابة من أصل 8. النتيجة مبنية على الأدلة الفعلية في أعلى 3 نتائج وتخص هذا المستند والأسئلة والنموذج فقط.",
        "", "### أسوأ حالة فشل", "", worst_text, "",
        "Q9 وQ10 غير قابلتين للإجابة من المستند عمدًا. Chroma ستعيد أقرب 3 مقاطع حتى لو لم تحتوِ على الإجابة؛ لذلك وجود نتائج لا يثبت وجود إجابة. يحتاج RAG إلى سياسة امتناع عند غياب دليل كافٍ.", "",
        "## Metadata", "", "كل Chunk يحمل source_file وpage_number وchunk_index وchunk_size_chars وoverlap_chars وchunking_method.",
        f"التحقق الفعلي: {meta_valid}/{meta_total} نتيجة مسترجعة احتوت اسم الملف ورقم الصفحة بشكل صحيح.", "",
        "## المفاهيم", "",
        "### Chunk", "الـ Chunk جزء من المستند يتحول إلى Vector مستقل. Embedding المستند كله قد يخلط موضوعات مختلفة في تمثيل واحد ويجعل الاسترجاع واسعًا بدل الجزء الذي يحتوي على الدليل.", "",
        "### Chunk Size", "القطعة الصغيرة جدًا قد تفصل الجملة عن تفسيرها أو تقطع الفكرة. والقطعة الكبيرة جدًا قد تجمع موضوعات غير مرتبطة فتقل دقة الاسترجاع. الحجم المتوسط نقطة بداية للاختبار وليس قاعدة ثابتة.", "",
        "### Chunk Overlap", "الـ Overlap يكرر جزءًا من نهاية القطعة السابقة في بداية التالية حتى لا تضيع الفكرة عند الحد الفاصل. زيادته بلا داعٍ تكرر المعلومات وتستهلك مساحة وقد تجعل أعلى النتائج متشابهة.", "",
        "### طرق التقطيع", "- Fixed character: عدد حروف ثابت وسهل القياس لكنه قد يقطع جملة أو فقرة.",
        "- Paragraph / sentence: يحافظ على حدود الفقرات والجمل لكن أحجام القطع تختلف.",
        "- Semantic: يبدأ قطعة جديدة عند انتقال المعنى باستخدام Embeddings أو نموذج لغوي؛ قد يحسن ترابط القطعة لكنه أعقد وأعلى تكلفة ويحتاج تقييمًا.", "",
        "### Metadata", "بيانات وصفية مرتبطة بكل Chunk مثل اسم الملف ورقم الصفحة. ده يسمح بإرجاع المصدر وعمل Citation بدل عرض نص مجهول المصدر.", "",
        "### الإجابة التي تنقسم على Chunkين", "لو السؤال يحتاج معلومة في قطعة وتكملتها في القطعة التالية فقد يرجع البحث واحدة فقط فتضيع علاقة مهمة أو يركب النموذج إجابة ناقصة. Overlap أو احترام حدود الفقرات أو استرجاع القطع المجاورة يقلل المشكلة لكنه لا يلغي الاختبار.", "",
        "## المنهجية والقيود", "- ثبتنا المصدر والأسئلة ونموذج Embedding وTop 3 ومقياس المسافة.",
        "- المتغيران هما حجم القطعة والتداخل فقط.",
        "- المقارنة الرقمية Fixed Character داخل كل صفحة. Paragraph / Sentence وSemantic موضحان نظريًا ولم يدخلا كمتغيرات إضافية.",
        "- مجموعات الأدلة وصفحاتها تم تحديدها يدويًا قبل التشغيل.",
        "- Cosine distance قيمة أقل تعني قربًا أكبر وليست Cosine Similarity.", "",
        "## إعادة التشغيل", "python -m pip install -r requirements.txt", "python W7D2_Chunking_Experiment.py", "",
        "النتائج التفصيلية في W7D2_chunking_comparison.csv. هذا التقرير اتولد من تشغيل التجربة الفعلي."
    ]
    REPORT.write_text("\n".join(report), encoding="utf-8")
    print("\n=== FINAL VERIFICATION ===")
    print(f"Comparison rows: {len(rows)} / 30")
    print(f"Valid source metadata: {meta_valid}/{meta_total}")
    print(f"Best configuration: {winner_name} ({winner['coverage']:.1%} evidence coverage)")
    print(f"CSV: {CSV_FILE}")
    print(f"Report: {REPORT}")
    print("[DONE] راجعي أسوأ حالة ومقتطفات النتائج قبل التسليم.")

if __name__ == "__main__":
    main()
