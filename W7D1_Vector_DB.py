from pathlib import Path

import chromadb


DB_PATH = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "phantoms_knowledge"
NOTES_FILE = Path(__file__).parent / "notes.md"

DOCUMENTS = [
    {"id": "programming_01", "topic": "programming", "text": "Python lists are mutable sequences that can store items of different types."},
    {"id": "programming_02", "topic": "programming", "text": "Git branches let developers work on features without affecting the main code."},
    {"id": "programming_03", "topic": "programming", "text": "A REST API exposes resources through HTTP methods like GET and POST."},
    {"id": "programming_04", "topic": "programming", "text": "SQL JOINs combine rows from two tables based on a related column."},
    {"id": "fitness_01", "topic": "fitness", "text": "Regular walking for thirty minutes a day improves heart health."},
    {"id": "fitness_02", "topic": "fitness", "text": "Drinking enough water helps maintain energy and concentration."},
    {"id": "fitness_03", "topic": "fitness", "text": "Sleep of seven to eight hours supports muscle recovery and memory."},
    {"id": "fitness_04", "topic": "fitness", "text": "Stretching before exercise can reduce the risk of muscle injuries."},
    {"id": "cooking_01", "topic": "cooking", "text": "To make rice fluffy, rinse it first and let it steam after cooking."},
    {"id": "cooking_02", "topic": "cooking", "text": "Letting dough rest allows the gluten to relax and makes bread softer."},
    {"id": "cooking_03", "topic": "cooking", "text": "Searing meat at high heat creates a flavorful brown crust."},
    {"id": "cooking_04", "topic": "cooking", "text": "Adding salt to pasta water seasons the noodles from the inside."},
]

QUERIES = [
    {"id": "Q1", "text": "How do I store different kinds of values together in Python?", "expected_topic": "programming"},
    {"id": "Q2", "text": "How can I avoid hurting myself when working out?", "expected_topic": "fitness"},
    {"id": "Q3", "text": "Why is my bread dense and hard?", "expected_topic": "cooking"},
    {"id": "Q4", "text": "What is the best way to merge data from two database tables?", "expected_topic": "programming"},
    {"id": "Q5", "text": "How do I deploy a rocket to Mars?", "expected_topic": None},
]

RESULTS_START = "<!-- RESULTS_START -->"
RESULTS_END = "<!-- RESULTS_END -->"


def create_collection():
    # بنفتح قاعدة البيانات الدائمة عشان البيانات تفضل موجودة بعد ما البرنامج يقفل
    client = chromadb.PersistentClient(path=str(DB_PATH))

    # بنستخدم الإعدادات الافتراضية للمقياس من غير ما نفرض مقياس مختلف
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # بنستخدم upsert عشان تشغيل الملف أكتر من مرة يحدث نفس المعرفات بدل ما يكررها
    collection.upsert(
        ids=[item["id"] for item in DOCUMENTS],
        documents=[item["text"] for item in DOCUMENTS],
        metadatas=[{"topic": item["topic"]} for item in DOCUMENTS],
    )

    return collection


def print_query_results(title, result):
    print(f"\n{title}")
    print("-" * 90)

    for index, (document_id, document, metadata, distance) in enumerate(
        zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ),
        start=1,
    ):
        print(f"{index}. ID: {document_id}")
        print(f"   Topic: {metadata['topic']}")
        print(f"   Document: {document}")
        print(f"   Distance: {distance:.6f}")


def get_first_result(result):
    return {
        "id": result["ids"][0][0],
        "document": result["documents"][0][0],
        "topic": result["metadatas"][0][0]["topic"],
        "distance": float(result["distances"][0][0]),
    }


def run_searches(collection):
    query_results = {}

    for item in QUERIES:
        result = collection.query(query_texts=[item["text"]], n_results=3)
        query_results[item["id"]] = result
        print_query_results(f"{item['id']} — {item['text']}", result)

    # بنعيد سؤال Q2 مع فلتر المجال عشان نقارن النتيجتين
    q2_filtered = collection.query(
        query_texts=[QUERIES[1]["text"]],
        n_results=3,
        where={"topic": "fitness"},
    )
    print_query_results("Q2 — نفس السؤال مع فلتر topic = fitness", q2_filtered)

    return query_results, q2_filtered


def build_threshold_analysis(query_results):
    relevant_distances = [
        get_first_result(query_results[query_id])["distance"]
        for query_id in ("Q1", "Q2", "Q3", "Q4")
    ]
    out_of_scope_distance = get_first_result(query_results["Q5"])["distance"]
    highest_relevant_distance = max(relevant_distances)

    if out_of_scope_distance > highest_relevant_distance:
        threshold = (highest_relevant_distance + out_of_scope_distance) / 2
        threshold_text = (
            f"المسافة المقترحة من التجربة هي **{threshold:.6f}**. "
            "القيمة دي منتصف المسافة بين أبعد أقرب نتيجة للأسئلة المرتبطة "
            "وأقرب نتيجة لسؤال خارج نطاق البيانات."
        )
        threshold_rule = (
            "لو أقل Distance للسؤال أكبر من Threshold المقترح، "
            "نعتبر إن مفيش نتيجة قريبة كفاية ونمتنع عن تقديم إجابة اعتمادًا على قاعدة البيانات."
        )
    else:
        threshold = None
        threshold_text = (
            "التجربة الحالية ماظهرتش فاصل واضح يسمح باختيار Threshold واحد "
            "يفصل كل الأسئلة المرتبطة عن السؤال الخارج عن النطاق. "
            "مينفعش نختار رقم ونقدمه كأنه حد موثوق."
        )
        threshold_rule = (
            "محتاجين بيانات واختبارات أكتر قبل اعتماد Threshold، "
            "ومينفعش نرفض أو نقبل السؤال اعتمادًا على رقم غير قادر على الفصل."
        )

    return {
        "relevant_distances": relevant_distances,
        "highest_relevant_distance": highest_relevant_distance,
        "out_of_scope_distance": out_of_scope_distance,
        "threshold": threshold,
        "threshold_text": threshold_text,
        "threshold_rule": threshold_rule,
    }


def build_results_section(collection, query_results, q2_filtered, analysis):
    lines = [
        RESULTS_START,
        "",
        "## نتائج التشغيل الفعلية",
        "",
        f"- عدد المستندات داخل الـCollection بعد التشغيل: **{collection.count()}**",
        "- عدد المستندات المطلوبة: **12**",
        "",
        "### أقرب نتيجة لكل سؤال",
        "",
        "| السؤال | أول نتيجة | المجال | المسافة | النتيجة متوافقة مع المجال المتوقع؟ |",
        "|---|---|---|---:|---|",
    ]

    for item in QUERIES:
        first = get_first_result(query_results[item["id"]])
        expected = item["expected_topic"]
        if expected is None:
            correctness = "سؤال خارج المجالات الموجودة"
        else:
            correctness = "نعم" if first["topic"] == expected else "لا"

        lines.append(
            f"| {item['id']} | {first['document']} | {first['topic']} | "
            f"{first['distance']:.6f} | {correctness} |"
        )

    lines.extend([
        "",
        "### مقارنة Q2 قبل وبعد الفلتر",
        "",
        "| الحالة | أول نتيجة | المجال | المسافة |",
        "|---|---|---|---:|",
    ])

    q2_unfiltered_first = get_first_result(query_results["Q2"])
    q2_filtered_first = get_first_result(q2_filtered)
    lines.append(
        f"| من غير فلتر | {q2_unfiltered_first['document']} | "
        f"{q2_unfiltered_first['topic']} | {q2_unfiltered_first['distance']:.6f} |"
    )
    lines.append(
        f"| مع فلتر fitness | {q2_filtered_first['document']} | "
        f"{q2_filtered_first['topic']} | {q2_filtered_first['distance']:.6f} |"
    )

    lines.extend([
        "",
        "الفلتر بيقصر البحث على المستندات اللي Metadata بتاعتها فيها "
        "topic = fitness، لكنه مش بيغير معنى السؤال نفسه.",
        "",
        "### مقارنة Q5 بالأسئلة المرتبطة",
        "",
        "| القياس | المسافة |",
        "|---|---:|",
    ])

    for query_id, distance in zip(("Q1", "Q2", "Q3", "Q4"), analysis["relevant_distances"]):
        lines.append(f"| أقرب نتيجة في {query_id} | {distance:.6f} |")

    lines.extend([
        f"| أبعد مسافة لأول نتيجة بين Q1 و Q4 | {analysis['highest_relevant_distance']:.6f} |",
        f"| أقرب نتيجة في Q5 | {analysis['out_of_scope_distance']:.6f} |",
        "",
        analysis["threshold_text"],
        "",
        analysis["threshold_rule"],
        "",
        "### أقرب 3 نتائج لسؤال Q5",
        "",
    ])

    q5 = query_results["Q5"]
    for index, (document, metadata, distance) in enumerate(
        zip(q5["documents"][0], q5["metadatas"][0], q5["distances"][0]),
        start=1,
    ):
        lines.append(
            f"{index}. **{metadata['topic']}** — {document} "
            f"(Distance: {float(distance):.6f})"
        )

    lines.extend([
        "",
        "### اختبار منع التكرار",
        "",
        "البرنامج بيستخدم upsert ومعرفات ثابتة للمستندات. "
        "شغلي الملف مرة تانية وتأكدي إن عدد المستندات فضل 12 بدل ما يزيد لـ24. "
        "العدد اللي فوق هو العدد الفعلي بعد التشغيل الحالي.",
        "",
        RESULTS_END,
    ])

    existing = NOTES_FILE.read_text(encoding="utf-8")
    start = existing.find(RESULTS_START)
    end = existing.find(RESULTS_END)
    if start == -1 or end == -1 or end < start:
        raise RuntimeError("علامات قسم النتائج مش موجودة في notes.md")

    updated = existing[:start] + "\n".join(lines) + existing[end + len(RESULTS_END):]
    NOTES_FILE.write_text(updated, encoding="utf-8")


def main():
    collection = create_collection()
    print("تم فتح قاعدة البيانات الدائمة")
    print(f"عدد المستندات بعد upsert: {collection.count()}")

    query_results, q2_filtered = run_searches(collection)
    analysis = build_threshold_analysis(query_results)

    print("\n=== تحليل Threshold ===")
    print(analysis["threshold_text"])
    print(analysis["threshold_rule"])

    build_results_section(collection, query_results, q2_filtered, analysis)

    print("\nتم تحديث النتائج الفعلية داخل notes.md")
    print("شغلي الملف مرة تانية وتأكدي إن عدد المستندات لسه 12")


if __name__ == "__main__":
    main()
