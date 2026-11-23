# W7D4 — RAG Pipeline Debugging

## الهدف

اختبار القدرة على تحديد مصدر الإجابة الخاطئة داخل RAG بدل الاكتفاء بملاحظة أن الإجابة غير صحيحة. الأعطال موزعة على Indexing وRetrieval وGeneration، وأول خطوة في التشخيص هي طباعة المقاطع المسترجعة قبل الحكم على النموذج.

## الملفات

- W7D3_RAG_Pipeline_BASELINE.py: نسخة محفوظة من ملف تنفيذ W7D3 الأصلي دون تغيير.
- W7D4_RAG_Debugging.py: مشغل التجارب المعزولة، ويعيد تطبيق مسار السؤال مع نقطة العطل المحددة.
- results/: ملف JSON وCSV مستقل لكل تجربة.
- indexes/: فهارس تجريبية معزولة.
- قاعدة W7D2 الأصلية لا يتم تعديلها.

كل تجربة تبدأ من نسخة W7D2 السليمة أو من فهرس تجريبي منفصل، وبالتالي يتم استرجاع الحالة السليمة تلقائيًا قبل الانتقال للتجربة التالية. لا يتم تعديل ملف D3 الأصلي أو قاعدة بياناته.

## متطلبات التشغيل

من جذر المستودع على Fedora:

~~~bash
cd ~/Projects/phantoms-ai-week7
source .venv/bin/activate
python -m py_compile W7D4/W7D3_RAG_Pipeline_BASELINE.py W7D4/W7D4_RAG_Debugging.py
ollama list
~~~

تأكد من وجود qwen2.5:3b، وملف data/source_document.pdf، وقاعدة chroma_db/w7d2_chunking التي تحتوي على Collection large_1600_250.

## تشغيل التجارب

كل أمر يشغّل الأسئلة العشرة نفسها ويحفظ JSON وCSV بعد كل سؤال. شغّل كل أمر منفصلًا وصوّر مخرجاته في الطرفية:

~~~bash
python W7D4/W7D4_RAG_Debugging.py --fault embedding 2>&1 | tee W7D4_embedding.log
python W7D4/W7D4_RAG_Debugging.py --fault chunk-small 2>&1 | tee W7D4_chunk-small.log
python W7D4/W7D4_RAG_Debugging.py --fault topk-1 2>&1 | tee W7D4_topk-1.log
python W7D4/W7D4_RAG_Debugging.py --fault topk-high 2>&1 | tee W7D4_topk-high.log
python W7D4/W7D4_RAG_Debugging.py --fault prompt 2>&1 | tee W7D4_prompt.log
python W7D4/W7D4_RAG_Debugging.py --fault metadata 2>&1 | tee W7D4_metadata.log
~~~

عدد جولات التشغيل ست؛ لأن اختبار Top-k المنخفض والمرتفع حالتان منفصلتان ضمن بند Top-k. كل جولة تشغّل الأسئلة العشرة نفسها. تجربة Chunk Size تعيد فهرسة التقرير بحجم 80 حرفًا وتداخل 10. وتجربة Metadata تنشئ نسخة منفصلة بمفاتيح Metadata متغيرة.

## جدول حقن الأعطال

| التجربة | إيه اللي اتغير؟ | العرض اللي ظهر | الطبقة والسبب | الحل |
|---|---|---|---|---|
| Embedding mismatch | استخدام sentence-transformers/all-MiniLM-L6-v2 وقت السؤال فقط بدل نموذج الفهرسة متعدد اللغات | يُسجل بعد التشغيل من نتائج الأسئلة العشرة | مقارنة نموذج السؤال بنموذج الفهرسة وفحص المقاطع المسترجعة قبل الإجابة | توحيد نموذج الـ Embedding بين الفهرسة والاستعلام وإعادة بناء الفهرس عند تغييره |
| Chunk size | إعادة فهرسة التقرير بحجم 80 حرفًا وتداخل 10 بدل 1600/250 | يُسجل بعد التشغيل من نتائج الأسئلة العشرة | Indexing: مقارنة عدد المقاطع ومحتواها وقدرتها على حفظ سياق الدليل | اختيار حجم وتداخل مناسبين لطبيعة المستند وقياس Recall |
| Top-k = 1 | إرجاع أقرب مقطع واحد | يُسجل بعد التشغيل من نتائج الأسئلة العشرة | Retrieval: فحص هل الدليل المطلوب موجود خارج أول نتيجة | رفع Top-k تدريجيًا وقياس Recall وجودة السياق |
| Top-k كبير | طلب 1000 نتيجة مع تقييد العدد الفعلي بحجم الفهرس | يُسجل بعد التشغيل من نتائج الأسئلة العشرة | Retrieval / Context: فحص الضوضاء والتكرار وحجم السياق | ضبط Top-k وإضافة reranking عند الحاجة |
| إزالة Grounding Prompt | السماح باستخدام المعرفة العامة والاستنتاج خارج Context | يُسجل بعد التشغيل خصوصًا في Q9 وQ10 | Generation: مقارنة ادعاءات الإجابة بما هو موجود فعلًا في المقاطع | إعادة تعليمات Grounding والامتناع والتحقق من الادعاءات |
| Metadata corruption | تغيير source_file إلى filename وpage_number إلى page | يُسجل بعد التشغيل من نتائج الأسئلة العشرة | Indexing / Metadata: فحص مفاتيح Metadata والقيم المفقودة | تثبيت Schema موحد والتحقق منه وقت الإدخال والاسترجاع |

**النتائج الفعلية لا تُفترض مسبقًا.** ملفات JSON وCSV وسجلات الطرفية الناتجة عن التشغيل هي المرجع لتسجيل العرض الذي ظهر فعلًا في كل سؤال. ظهور إجابة مولدة لا يعني أنها صحيحة، كما أن اختلاف جودة النتائج بين سؤال وآخر متوقع.

## Checklist تشخيص إجابة RAG خاطئة

1. اطبع نصوص المقاطع المسترجعة قبل الحكم على إجابة النموذج.
2. تحقق من قاعدة البيانات والـ Collection المستخدمة وعدد السجلات.
3. تحقق من تطابق نموذج Embedding بين الفهرسة والاستعلام.
4. افحص Metadata: اسم المصدر ورقم الصفحة ومعرف المقطع والقيم المفقودة.
5. قيّم Retrieval: هل الدليل موجود في المستند؟ وهل ظهر في النتائج؟ وهل Top-k مناسب؟
6. افحص Chunking: هل الدليل اتقطع بين مقطعين؟ وهل الحجم والتداخل مناسبان؟
7. لو الدليل موجود في Context، افحص Generation: هل النموذج تجاهله أو حرّفه أو أضاف ادعاءً؟
8. راجع System Prompt وإعدادات النموذج، خصوصًا Grounding والامتناع ودرجة الحرارة.
9. أعد السؤال نفسه بعد تغيير متغير واحد فقط وسجل الفرق قبل اعتماد الحل.

## قراءة النتائج

كل تجربة تنتج:
- results/<fault>.json: الإعدادات والإجابات والمقاطع المسترجعة والصفحات والمسافات وسلامة Metadata والأخطاء.
- results/<fault>.csv: ملخص جدولي للأسئلة والإجابات وعدد النتائج والصفحات.

GENERATED_REQUIRES_REVIEW معناها أن النموذج ولّد إجابة فقط، ولا تعني أنها صحيحة. تتم مقارنة الإجابة بالنصوص المسترجعة وبملف PDF الأصلي.

## لقطات التيرمنال

1. فحص صياغة الملفين وظهور قاعدة W7D2 وعدد المقاطع.
2. تجربة Embedding mismatch مع ظهور نموذج السؤال المختلف ونصوص النتائج.
3. تجربة Chunk size مع ظهور إعادة الفهرسة وعدد المقاطع الجديدة.
4. تجربة Top-k = 1 ثم Top-k الكبير وعدد النتائج المعادة.
5. تجربة Prompt غير المقيد وإجابات Q9 وQ10.
6. تجربة Metadata المعطوبة وظهور الحقول المفقودة.
7. ملخص كل تجربة ووجود ملفات JSON وCSV.

## رابط المستودع

https://github.com/inspire2029-sudo/phantoms-ai-week7

## المصادر

- Redis — RAG debugging guide: https://redis.io/blog/rag-debugging-guide-retrieval-errors/
- Latitude — LLM failure modes and root-cause analysis: https://latitude.so/blog/debug-rag-pipeline-wrong-answers
- FastEmbed — Supported embedding models: https://qdrant.github.io/fastembed/examples/Supported_Models/
