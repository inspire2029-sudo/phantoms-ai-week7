# W7D4 — RAG Pipeline Debugging

## الهدف

اختبار القدرة على تحديد مصدر الإجابة الخاطئة داخل RAG بدل الاكتفاء بملاحظة أن الإجابة غير صحيحة. الأعطال موزعة على Indexing وRetrieval وGeneration، وأول خطوة في التشخيص هي طباعة المقاطع المسترجعة قبل الحكم على النموذج.

## الملفات

- W7D4_RAG_Debugging.py: برنامج حقن الأعطال وتشغيل نفس الأسئلة العشرة.
- results/: ملف JSON وCSV مستقل لكل تجربة.
- indexes/: فهارس تجريبية معزولة.
- قاعدة W7D2 الأصلية لا يتم تعديلها.

## متطلبات التشغيل

من جذر المستودع على Fedora:

~~~bash
cd ~/Projects/phantoms-ai-week7
source .venv/bin/activate
python -m py_compile W7D4/W7D4_RAG_Debugging.py
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

كل تجربة مستقلة. لا يتم تعديل قاعدة W7D2 الأصلية. فهرس Chunk الصغير ونسخة Metadata المعطوبة بيتخزنوا داخل W7D4/indexes فقط.

## جدول حقن الأعطال

| التجربة | التغيير | الطبقة الأساسية | العرض الذي يتم فحصه | طريقة تحديد السبب | الحل |
|---|---|---|---|---|---|
| Embedding mismatch | استخدام sentence-transformers/all-MiniLM-L6-v2 وقت السؤال فقط مع إبقاء فهرس W7D2 كما هو | توافق الفهرسة والاستعلام | تراجع ملاءمة النتائج أو خطأ أبعاد لو اختلفت الأبعاد | مقارنة نموذج الفهرسة بنموذج السؤال وفحص نصوص النتائج والمسافات قبل الإجابة | استخدام نفس نموذج Embedding وإصداره في الفهرسة والاستعلام وإعادة بناء الفهرس عند تغييره |
| Chunk size | إعادة فهرسة التقرير بحجم 80 حرفًا وتداخل 10 بدل 1600/250 | Indexing | تقطيع الدليل وفصل عناصر الإجابة أو سياقها | مقارنة عدد المقاطع وأحجامها ونصوص النتائج وصفحاتها | قياس حجم مناسب والحفاظ على حدود الفقرات وضبط التداخل |
| Top-k = 1 | إرجاع أقرب مقطع واحد | Retrieval | نقص الأدلة متعددة الأجزاء أو إجابة جزئية أو امتناع غير ضروري | طباعة عدد النتائج ونص المقطع الأول ومقارنته بالدليل | رفع Top-k تدريجيًا وقياس Recall وجودة السياق |
| Top-k كبير | طلب 1000 نتيجة مع تقييد العدد الفعلي بحجم الفهرس | Retrieval | سياق أطول قد يضيف ضوضاء وتكرارًا ويضعف تركيز التوليد | فحص عدد النتائج وترتيبها وتنوع الصفحات وموضع الدليل | ضبط Top-k وإضافة reranking عند الحاجة |
| إزالة Grounding Prompt | السماح باستخدام المعرفة العامة والاستنتاج خارج Context | Generation | احتمال ظهور إجابة واثقة أو تفاصيل غير مدعومة خصوصًا Q9 وQ10 | التأكد أن الادعاء غير موجود في النصوص المسترجعة ثم مقارنة الإجابة | إعادة تعليمات Grounding والامتناع والتحقق من الادعاءات |
| Metadata corruption | تغيير source_file إلى filename وpage_number إلى page | Indexing / Metadata | فقد اسم الملف ورقم الصفحة رغم بقاء النص المسترجع | طباعة مفاتيح Metadata وملاحظة MISSING_SOURCE_FILE وMISSING_PAGE_NUMBER | تثبيت Schema موحد والتحقق منه وقت الإدخال والاسترجاع |

التجربة العملية فيها ست جولات لأن اختبار Top-k المنخفض والمرتفع تجربتان منفصلتان، وكل جولة تشغّل الأسئلة العشرة نفسها.

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

1. فحص صياغة البرنامج وظهور قاعدة W7D2 وعدد المقاطع.
2. تجربة Embedding mismatch مع ظهور نموذج السؤال المختلف ونصوص النتائج.
3. تجربة Chunk size مع ظهور إعادة الفهرسة وعدد المقاطع الجديدة.
4. تجربة Top-k = 1 ثم Top-k الكبير وعدد النتائج المعادة.
5. تجربة Prompt غير المقيد وإجابات Q9 وQ10.
6. تجربة Metadata المعطوبة وظهور الحقول المفقودة.
7. ملخص كل تجربة ووجود ملفات JSON وCSV.

## رابط المستودع

https://github.com/inspire2029-sudo/phantoms-ai-week7

## المصادر

- Redis — RAG Debugging Guide: Retrieval Errors: https://redis.io/blog/rag-debugging-guide-retrieval-errors/
- Latitude — How to debug a RAG pipeline that returns wrong answers: https://latitude.so/blog/how-to-debug-a-rag-pipeline-that-returns-wrong-answers/
