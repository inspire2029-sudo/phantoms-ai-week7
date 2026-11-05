"""
PHANTOMS AI — Week 7
W7D1 — Vector Database باستخدام Chroma

الهدف من التاسك
في الأسبوع السادس حولت الجمل إلى Embeddings واستخدمت التشابه عشان أبحث عن الجمل الأقرب لسؤال جديد
في التاسك ده هنستخدم Chroma عشان نخزن المتجهات وندير البحث الدلالي

العلاقة بالتاسكات القديمة
في الأسبوع التاني اشتغلت على SQL والجداول والعلاقات والاستعلامات زي JOIN
وفي الأسبوع السادس استخدمت sentence-transformers/all-MiniLM-L6-v2 عشان أحول النصوص إلى Embeddings
وبعدها حسبت Cosine Similarity وEuclidean Distance وعملت Mini Semantic Search
هنا Chroma هتدير التخزين والبحث بدل ما أرتب المتجهات بنفسي

الفرق بين SQL Database و Vector Database
SQL بتنظم البيانات في جداول وصفوف وأعمدة وبتناسب البحث بالشروط والقيم والعلاقات
Vector Database بتخزن المتجهات ومعاها Metadata وبتبحث عن العناصر الأقرب لمعنى السؤال
ده مش معناه إن واحدة بديل كامل للتانية لأن كل واحدة ليها استخدام مختلف وممكن النظام يستخدم الاتنين مع بعض

مقياس المسافة
الـCollection الافتراضية في Chroma بتستخدم l2 في Single Node وده Squared Euclidean Distance
كل ما المسافة تقل النتيجة تكون أقرب حسب المقياس
المصدر الرسمي https://docs.trychroma.com/docs/collections/configure

ليه Q5 مهم
Q5 بيسأل عن إطلاق صاروخ للمريخ بينما البيانات الموجودة عن البرمجة واللياقة والطبخ فقط
Vector Search بيرجع أقرب نتائج موجودة حتى لو السؤال بعيد عن محتوى قاعدة البيانات
وجود نتيجة في أول الترتيب مش دليل إن فيه معلومة مناسبة للإجابة
في أنظمة RAG لازم نراجع مدى قرب النتائج قبل ما نبعت السياق للموديل
لو المسافة كبيرة أو مفيش دليل كفاية ممكن النظام يمتنع عن استخدام النتيجة بدل ما يدي إجابة مبنية على سياق غير مرتبط

Threshold
مش هنختار Threshold من رقم منتشر لأن المسافات بتختلف حسب موديل Embeddings ومقياس المسافة والبيانات
هنقارن أقرب مسافة في Q5 بأقرب المسافات للأسئلة المرتبطة من Q1 إلى Q4
لو ظهر فاصل واضح هنقترح قيمة في منتصف الفاصل
لو مظهرش فاصل واضح هنسجل إن التجربة مش كفاية لاختيار Threshold موثوق
الأرقام الفعلية بتظهر في مخرجات التشغيل تحت

منع التكرار
بنستخدم upsert ومعرفات ثابتة عشان تشغيل الملف أكتر من مرة يحدث نفس السجلات بدل ما يكررها
المفروض عدد المستندات يفضل 12 بعد كل تشغيل

التشغيل
python -m pip install chromadb
python W7D1_Vector_DB.py

المصادر
Chroma Getting Started: https://docs.trychroma.com/docs/overview/getting-started
Chroma Guides: https://docs.trychroma.com/guides
Chroma Collection Configuration: https://docs.trychroma.com/docs/collections/configure
Cyclr Vector Databases: https://cyclr.com/resources/ai/understanding-vector-databases-a-deep-dive-with-pinecone

ملاحظة
النتائج والمسافات هنا بتتولد من التشغيل الفعلي ومش مكتوبة مسبقا
"""

from pathlib import Path

import chromadb


DB_PATH = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "phantoms_knowledge"

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


def create_collection():
    # بنفتح قاعدة البيانات الدائمة عشان البيانات تفضل موجودة بعد ما البرنامج يقفل
    client = chromadb.PersistentClient(path=str(DB_PATH))

    # بننشئ الـCollection لو مش موجودة وبنستخدم إعدادات Chroma الافتراضية
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # بنستخدم upsert عشان نفس المعرفات تتحدث بدل ما تتكرر
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

    # بنعيد Q2 مع فلتر المجال عشان نقارن النتيجتين
    q2_filtered = collection.query(
        query_texts=[QUERIES[1]["text"]],
        n_results=3,
        where={"topic": "fitness"},
    )
    print_query_results("Q2 — نفس السؤال مع فلتر topic = fitness", q2_filtered)

    return query_results, q2_filtered


def analyze_threshold(query_results):
    # دي أقرب مسافة لأول نتيجة في الأسئلة اللي ليها مجال متوقع
    relevant_distances = [
        get_first_result(query_results[query_id])["distance"]
        for query_id in ("Q1", "Q2", "Q3", "Q4")
    ]

    # ودي أقرب مسافة للسؤال اللي خارج المجالات الموجودة
    out_of_scope_distance = get_first_result(query_results["Q5"])["distance"]
    highest_relevant_distance = max(relevant_distances)

    print("\n=== مقارنة المسافات ===")
    for query_id, distance in zip(("Q1", "Q2", "Q3", "Q4"), relevant_distances):
        print(f"أقرب مسافة في {query_id}: {distance:.6f}")

    print(f"أبعد مسافة بين الأسئلة المرتبطة: {highest_relevant_distance:.6f}")
    print(f"أقرب مسافة في Q5: {out_of_scope_distance:.6f}")

    print("\n=== Threshold Analysis ===")
    if out_of_scope_distance > highest_relevant_distance:
        threshold = (highest_relevant_distance + out_of_scope_distance) / 2
        print(f"Threshold المقترح من التجربة: {threshold:.6f}")
        print(
            "لو أقل Distance للسؤال أكبر من Threshold المقترح "
            "نعتبر إن مفيش نتيجة قريبة كفاية ونمتنع عن استخدام السياق"
        )
    else:
        print(
            "التجربة ماظهرتش فاصل واضح يسمح باختيار Threshold موثوق "
            "محتاجين أسئلة وبيانات أكتر قبل اعتماد حد فاصل"
        )


def print_comparison(query_results, q2_filtered):
    print("\n=== مقارنة أول نتيجة لكل سؤال ===")
    print(f"{'Query':<10} {'Topic':<15} {'Distance':<15} Expected")
    print("-" * 60)

    for item in QUERIES:
        first = get_first_result(query_results[item["id"]])
        expected = item["expected_topic"] or "out of scope"
        correctness = (
            "خارج النطاق"
            if item["expected_topic"] is None
            else ("صحيح" if first["topic"] == item["expected_topic"] else "غير متوافق")
        )
        print(
            f"{item['id']:<10} {first['topic']:<15} "
            f"{first['distance']:<15.6f} {expected} — {correctness}"
        )

    q2_unfiltered = get_first_result(query_results["Q2"])
    q2_filtered_first = get_first_result(q2_filtered)

    print("\n=== Q2: بدون فلتر مقابل فلتر fitness ===")
    print(
        f"بدون فلتر: {q2_unfiltered['topic']} | "
        f"{q2_unfiltered['document']} | {q2_unfiltered['distance']:.6f}"
    )
    print(
        f"مع الفلتر: {q2_filtered_first['topic']} | "
        f"{q2_filtered_first['document']} | {q2_filtered_first['distance']:.6f}"
    )


def main():
    collection = create_collection()

    print("تم فتح قاعدة البيانات الدائمة")
    print(f"عدد المستندات بعد upsert: {collection.count()}")
    print("المفروض العدد يفضل 12 حتى لو شغلنا الملف أكتر من مرة")

    query_results, q2_filtered = run_searches(collection)
    print_comparison(query_results, q2_filtered)
    analyze_threshold(query_results)

    print("\n=== ملخص التجربة ===")
    print(f"عدد المستندات النهائي: {collection.count()}")
    print("لو العدد 12 بعد تشغيل الملف مرتين يبقى upsert منع تكرار السجلات")


if __name__ == "__main__":
    main()
