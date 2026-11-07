# PHANTOMS AI — Week 7 / Day 2
## تجربة Chunking وتأثيره على جودة Retrieval في RAG

> **نوع المهمة:** تجربة عملية ومقارنة استرجاع  
> **المستند المستخدم:** `SGDS_SWARM_Engineering_Report_Civil_Edition(4) (1).pdf`  
> **عدد صفحات المستند المذكور في بيانات المشروع:** 53 صفحة  
> **قاعدة البيانات:** ChromaDB  
> **نموذج التضمين:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`  
> **عدد الأسئلة:** 10 أسئلة × 3 إعدادات = 30 عملية Retrieval

---

## فهرس المحتويات

1. [أنا عايزة أختبر إيه؟](#1-أنا-عايزة-أختبر-إيه)
2. [اختيار المستند](#2-اختيار-المستند)
3. [المفاهيم اللي بنيت عليها التجربة](#3-المفاهيم-اللي-بنيت-عليها-التجربة)
4. [الأسئلة والإجابات المرجعية](#4-الأسئلة-والإجابات-المرجعية)
5. [إعدادات الـ Chunking](#5-إعدادات-الـ-chunking)
6. [الـ Metadata](#6-الـ-metadata)
7. [طريقة التنفيذ](#7-طريقة-التنفيذ)
8. [جدول النتائج](#8-جدول-النتائج)
9. [تحليل النتائج والاستنتاج](#9-تحليل-النتائج-والاستنتاج)
10. [حدود التجربة](#10-حدود-التجربة)
11. [الكود الكامل](#11-الكود-الكامل)
12. [طريقة التشغيل](#12-طريقة-التشغيل)

---

## 1. أنا عايزة أختبر إيه؟

أنا هنا مش هدفي أقسم المستند لأجزاء وخلاص؛ أنا عايزة أقيس تأثير حجم كل جزء، ومقدار التداخل بين الأجزاء، على قدرة نظام RAG إنه يرجّع المعلومة الصح لما أسأله سؤال.

الفكرة إن نفس السؤال ممكن تكون إجابته موجودة في فقرة واحدة، أو متوزعة على أكتر من فقرة أو صفحة. عشان كده هقارن بين 3 إعدادات مختلفة، مع تثبيت نفس المستند ونفس الأسئلة ونفس نموذج الـ Embedding ونفس عدد النتائج المسترجعة.

هراجع في كل سؤال:

- هل الدليل المطلوب ظهر فعلًا في أول 3 نتائج؟
- لو الإجابة محتاجة أكتر من موضع، هل كل أجزاء الدليل ظهرت ولا جزء منها بس؟
- الدليل ظهر في أنهي نتيجة: الأولى ولا التانية ولا التالتة؟
- هل كل نتيجة رجّعت اسم الملف ورقم الصفحة في الـ Metadata؟
- في السؤال اللي إجابته مش موجودة، هل هقدر أميّز غياب الإجابة عن مجرد ظهور مقاطع قريبة في المعنى؟

**مهم:** مش هاعتبر إن ظهور 3 نتائج معناه إن الإجابة موجودة. لازم أراجع محتوى النتائج مقابل الدليل المرجعي المحدد لكل سؤال.

## 2. اختيار المستند

اخترت تقرير `SGDS_SWARM_Engineering_Report_Civil_Edition(4) (1).pdf` لأنه مستند هندسي حقيقي ومتعدد الأقسام، وفيه معلومات عن معمارية النظام، والعتاد، والذكاء الاصطناعي، وإدارة البيانات، والاتصالات، والأمن، والسلامة، وخطة التنفيذ.

اخترته تحديدًا لأن فيه أسئلة محتاجة ربط أكتر من معلومة، مش مجرد البحث عن كلمة موجودة في فقرة واحدة. وده مناسب لاختبار هل الـ Chunking بيحافظ على السياق ولا بيفصل أجزاء الإجابة عن بعضها.

السكربت بيتأكد وقت التشغيل إن ملف الـ PDF موجود، وإنه مكوّن من 5 صفحات على الأقل، وإن النص قابل للاستخراج من 5 صفحات على الأقل. أرقام الصفحات المستخدمة في التقييم هي أرقام صفحات ملف الـ PDF، وبتبدأ من 1.

## 3. المفاهيم اللي بنيت عليها التجربة

### يعني إيه Chunk؟

الـ Chunk هو جزء من المستند بتعامل معاه كوحدة مستقلة وقت الفهرسة والاسترجاع. بدل ما أحوّل التقرير كله لـ Embedding واحد، بقسمه لأجزاء، وكل جزء بياخد تمثيل رقمي مستقل يساعدني أسترجعه لما ييجي سؤال متعلق بمحتواه.

### ليه مش بعمل Embedding للمستند كله مرة واحدة؟

لأن المستند ممكن يحتوي على موضوعات مختلفة جدًا. لو حولته كله لمتجه واحد، ممكن تمثيل موضوع البطارية يتداخل مع الأمن والاتصالات والذكاء الاصطناعي، وساعتها مش هقدر أحدد الجزء اللي فيه الدليل بدقة كفاية.

تقسيم المستند بيسمحلي أسترجع أجزاء محددة، وأربط كل نتيجة بمصدرها ورقم صفحتها.

### Chunk Size

ده حجم الجزء اللي بقسم له النص. في التجربة دي القياس بالحروف، مش بعدد الـ Tokens.

- **الحجم الصغير جدًا:** ممكن يحافظ على تركيز الجزء، لكنه ممكن يفصل السؤال عن تكملة إجابته أو يقطع جملة في نصها.
- **الحجم الكبير جدًا:** ممكن يحتفظ بسياق أوسع، لكنه ممكن يجمع معلومات وموضوعات مختلفة في نفس الجزء، وده يقلل دقة الاسترجاع.
- **الحجم المتوسط:** نقطة وسط بين الاحتفاظ بالسياق وعدم إدخال كلام كتير ملوش علاقة بالسؤال. لكنه مش رقم مثالي لكل المستندات.

### Chunk Overlap

الـ Overlap هو عدد الحروف اللي بكررها من نهاية الجزء السابق في بداية الجزء اللي بعده.

أنا بستخدمه عشان أقلل احتمال إن معلومة تقع على الحد الفاصل بين جزأين فتتقطع. لكن لو زودته زيادة عن اللزوم، هكرر محتوى كتير في قاعدة البيانات، وممكن أول 3 نتائج تبقى شبه بعض بدل ما تعرض أدلة مختلفة.

### طرق التقطيع

| الطريقة | بتشتغل إزاي؟ | الميزة | العيب |
|---|---|---|---|
| Fixed-character | بتقسم النص حسب عدد حروف ثابت | بسيطة وسهلة المقارنة وإعادة التنفيذ | ممكن تقطع جملة أو فقرة في نص المعنى |
| Paragraph / Sentence-aware | بتحاول تحافظ على حدود الفقرات والجمل | بتحافظ على وحدات لغوية أوضح | أحجام الأجزاء مش ثابتة، والفقرات الطويلة محتاجة معالجة |
| Semantic chunking | بتبدأ جزء جديد لما الموضوع أو المعنى يتغير | ممكن تحافظ على وحدة الفكرة | أعقد وأعلى تكلفة، وبتعتمد على جودة نموذج قياس التشابه |

في الـ Benchmark ده استخدمت **Fixed-character** في الإعدادات الثلاثة، عشان أعزل تأثير الحجم والـ Overlap، وما أدخلش طريقة تقطيع تالتة كمتغير إضافي يغيّر نتيجة المقارنة.

## 4. الأسئلة والإجابات المرجعية

جهزت 10 أسئلة من نفس المستند. الأسئلة من Q1 إلى Q8 ليها إجابات في المصدر، والأسئلة Q9 وQ10 متعمدة تكون إجابتها مش موجودة في المستند.

وفيه أكتر من سؤال محتاج جمع دليل من صفحات مختلفة؛ وده مقصود عشان أختبر استرجاع الإجابة المركبة، مش مجرد استرجاع فقرة واحدة.

| ID | السؤال | الإجابة المرجعية المختصرة | صفحات الدليل |
|---|---|---|---|
| Q1 | ما الدور الذي تمثله كل طائرة داخل منظومة SGDS SWARM؟ | كل طائرة بتمثل عقدة استشعار ومعالجة، بتنّفذ الطيران وتستخلص المعلومات وتقيّم حالتها التشغيلية وتتبادل بيانات مختصرة وتتخذ قرارات محلية ضمن السياسات المحددة. | 3 |
| Q2 | ما الطبقات الوظيفية العشر للمنظومة وما الطبقتان العرضيتان اللتان تغطيانها؟ | الطبقات الوظيفية هي Airframe / Power / Mechanics وSensors / Payload وFlight Control وNavigation / Localization وCommunications وData Management وAI / Computer Vision وEvent & Decision Engine وFleet Management وUser Interface / Command Center. والطبقتان العرضيتان هما Cybersecurity وSafety / Reliability. | 8 و9 |
| Q3 | ليه التقرير مش بيعتمد على عتبة بطارية ثابتة، وإيه المدخلات المستخدمة لتقدير زمن المهمة الآمن؟ | لأن نسبة البطارية مش بتمثل نفس الاحتياطي في كل الظروف؛ الاستهلاك بيتغير حسب الحمولة والطقس والمسافة وحالة البطارية. التقدير بيعتمد على الطاقة المتبقية وطاقة العودة واحتياطي السلامة وتأثير الطقس وصحة البطارية واستهلاك الحمولة. | 13 و14 |
| Q4 | إيه المقاييس التشغيلية اللي التقرير بيوصي بتسجيلها بجانب mAP وليه؟ | بيوصي بتسجيل False Alerts/hour وMiss Rate وFalse Alarms per Operational Hour وInference Latency وFPS وWatt/Inference وThermal Throttling Rate؛ لأن mAP لوحده مش بيكشف الإنذارات الكاذبة أو تكلفة وسرعة التشغيل على العتاد. | 24 و25 |
| Q5 | النظام بيتعامل إزاي مع تراجع أداء إصدار نموذج جديد، وإزاي بيراقب تغير البيانات؟ | بيرصد Health Monitor تراجع الأداء وبعدها ينفذ Rollback للإصدار السابق. وبيتابع Data Drift عن طريق Average confidence وUnknown detections وFalse alarms وScene distribution وImage quality، وبعد اكتشاف الانحراف بيطلب Model Review Required. | 26 و27 |
| Q6 | إيه الإجراءات المطلوبة لإدارة مفاتيح وحدات الأسطول وتأمين تحديثات البرامج؟ | لازم تتحدد مسؤولية إنشاء المفاتيح وتوزيعها وتخزينها وصلاحيات الوصول وتدويرها وإلغائها والتعامل مع فقدان الجهاز. والتحديث الآمن بيعدي على Signed Package ثم Signature Verification والتوافق والتثبيت وفحص الصحة، مع Rollback لو حصل فشل. | 32 |
| Q7 | إيه المبدأ اللي لازم يحكم تصرف الطائرة عند فقدان اتصال الشبكة؟ | فقدان الشبكة مش معناه فقدان الطائرة؛ لازم كل طائرة تحتفظ بمنطق سلامة محلي وما تستناش أوامر الخادم وقت انقطاع الاتصال. | 44 |
| Q8 | إيه التسليمات المطلوبة من فريق الذكاء الاصطناعي غير ملف model.pt؟ | تشمل Dataset specification وDataset version وLabel policy وTraining pipeline وEvaluation report وModel artifact وTensorRT artifact وBenchmark وInference service وEvent API وModel card وFailure cases وDrift monitor وRollback version. | 49 |
| Q9 | إيه قيمة mAP اللي النموذج حققها فعليًا في اختبار ميداني موثق على جهاز Jetson المستهدف؟ | المعلومة دي مش موجودة في المستند. التقرير بيذكر مقاييس مطلوب قياسها، لكنه مش بيقدم قيمة mAP موثقة لاختبار ميداني على الجهاز المستهدف. | غير موجود |
| Q10 | كام رحلة اختبار ميداني مكتملة اتنفذت فعليًا، وإيه متوسط زمن الطيران المقاس بالدقائق؟ | المعلومة دي مش موجودة في المستند؛ مفيش عدد رحلات مكتملة أو متوسط زمن طيران مقاس بالدقائق. | غير موجود |

**ملاحظة على الأسئلة متعددة الأدلة:** Q2 وQ3 وQ5 وQ6 وQ8 محتاجين تجميع دليل من أكتر من موضع أو مجموعة معلومات. في التقييم، الإجابة المركبة مش هتتحسب مكتملة لمجرد ظهور جزء واحد من الدليل.

## 5. إعدادات الـ Chunking

ثبتّ نفس المصدر والأسئلة ونموذج الـ Embedding وعمق الاسترجاع، وغيّرت حجم الجزء والـ Overlap بس.

| الإعداد | Chunk Size | Overlap | الوصف |
|---|---:|---:|---|
| Small | 180 حرف | 30 حرف | صغير جدًا |
| Medium | 700 حرف | 120 حرف | متوسط |
| Large | 1600 حرف | 250 حرف | كبير جدًا |

كل صفحة بتتقطع بشكل مستقل؛ يعني السكربت مش بيجمع آخر صفحة مع أول الصفحة اللي بعدها في Chunk واحد. ده بيساعدني أحافظ على رقم صفحة واضح لكل جزء، وبالتالي أقدر أرجع لموضع الدليل في الـ PDF.

## 6. الـ Metadata

كل Chunk بيتخزن معاه Metadata تساعدني أعرف النص ده جاي منين وإعدادات تقطيعه كانت إيه.

| اسم الحقل | مثال | استخدامه |
|---|---|---|
| `source_file` | `SGDS_SWARM_Engineering_Report_Civil_Edition(4) (1).pdf` | اسم ملف المصدر |
| `page_number` | `14` | رقم صفحة الـ PDF |
| `chunk_index` | `2` | ترتيب الجزء داخل الصفحة |
| `chunk_size_chars` | `700` | حجم الجزء بالحروف |
| `overlap_chars` | `120` | مقدار التداخل بالحروف |
| `chunking_method` | `fixed_character_per_page` | طريقة التقطيع |

بالنسبة لي، وجود اسم الملف ورقم الصفحة مش تفصيلة شكلية؛ ده اللي بيخليني أقدر أراجع الدليل وأوثق مصدره بدل ما أعتمد على نص مسترجع من غير أصل واضح.

## 7. طريقة التنفيذ

خطوات التجربة في السكربت:

1. قراءة ملف الـ PDF صفحة صفحة واستخراج النص.
2. التأكد إن المستند 5 صفحات على الأقل وإن فيه 5 صفحات على الأقل نصها قابل للاستخراج.
3. تقسيم كل صفحة حسب الإعداد الحالي، مع إضافة الـ Metadata لكل Chunk.
4. إنشاء Collection مستقلة في ChromaDB لكل إعداد.
5. إنشاء Embeddings للنصوص باستخدام `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
6. تخزين الـ Embeddings والنصوص والـ Metadata في ChromaDB باستخدام Cosine distance.
7. تحويل كل سؤال إلى Embedding واسترجاع أعلى 3 نتائج.
8. مقارنة النتائج بمواضع الدليل المرجعية المحددة لكل سؤال.
9. تسجيل ترتيب ظهور الدليل، وصفحات النتائج الثلاث، وقيم المسافة، ومعرفات الـ Chunks، والتحقق من الـ Metadata.
10. حفظ الـ 30 تجربة في ملف CSV مساعد للتحليل والمراجعة.

## 8. النتائج الفعلية

تم تنفيذ المقارنة بالفعل على المستند المحدد، والنتائج التالية هي ناتج التشغيل المسجل للتجربة. **لم تتم إعادة تشغيل التجربة أثناء إغلاق التوثيق**؛ تم توثيق القياسات الموجودة كما هي.

### ملخص المقارنة

| الإعداد | عدد الـ Chunks | الأدلة المسترجعة من 78 | Micro Evidence Coverage | متوسط تغطية السؤال | أسئلة مكتملة | جزئية | فاشلة |
|---|---:|---:|---:|---:|---:|---:|---:|
| Small 180/30 | 429 | 22/78 | 28.2% | 31.1% | 1/8 | 5/8 | 2/8 |
| Medium 700/120 | 120 | 42/78 | 53.8% | 51.5% | 3/8 | 2/8 | 3/8 |
| Large 1600/250 | 62 | 43/78 | 55.1% | 55.8% | 4/8 | 1/8 | 3/8 |

تمثل **Micro Evidence Coverage** نسبة كل الـ Anchors التي ظهرت في Top 3 من إجمالي 78 Anchor. أما **متوسط تغطية السؤال** فهو متوسط نسبة الأدلة المسترجعة داخل كل سؤال على حدة؛ لذلك قد يختلف عن النسبة الإجمالية.

### تغطية الأدلة لكل سؤال

كل قيمة تمثل عدد الـ Anchors التي ظهرت في Top 3 من إجمالي Anchors السؤال:

| السؤال | Small 180/30 | Medium 700/120 | Large 1600/250 |
|---|---:|---:|---:|
| Q1 | 0/6 | 0/6 | 6/6 |
| Q2 | 2/12 | 10/12 | 0/12 |
| Q3 | 6/13 | 13/13 | 6/13 |
| Q4 | 0/7 | 2/7 | 0/7 |
| Q5 | 3/9 | 0/9 | 0/9 |
| Q6 | 5/13 | 13/13 | 13/13 |
| Q7 | 4/4 | 4/4 | 4/4 |
| Q8 | 2/14 | 0/14 | 14/14 |

### التحقق من الـ Metadata

- عدد صفوف المقارنة: **30/30** (10 أسئلة × 3 إعدادات).
- عدد نتائج Top 3 التي تم التحقق من بياناتها: **90/90**.
- التحقق شمل اسم ملف المصدر ورقم صفحة صحيحًا موجبًا في كل نتيجة.

### Q9 وQ10

السكربت يصنف السؤالين كـ `ABSENCE_CHECK` ويستبعدهما من مقام تغطية الأدلة؛ لأنهما لا يملكان Anchors إيجابية قابلة للبحث. عدم ظهور إجابة في Top 3 **ليس وحده إثباتًا** أن المعلومة غير موجودة في كامل المستند. لذلك يظل الحكم على غياب قيمة mAP الميدانية وعدد الرحلات ومتوسط زمن الطيران حكمًا مبنيًا على مراجعة المصدر، وليس على نتيجة الاسترجاع وحدها.

## 9. تحليل النتائج والاستنتاج

### ماذا تقول النتائج؟

1. **Large حقق أعلى تغطية إجمالية بفارق محدود:** استرجع 43 من 78 Anchor (55.1%) مقابل 42 من 78 (53.8%) لـ Medium. الفارق Anchor واحد فقط؛ لذلك لا يكفي وحده لإثبات تفوق عام أو حاسم.
2. **Large حقق أكبر عدد من الأسئلة المكتملة:** 4 من 8، مقابل 3 من 8 لـ Medium و1 من 8 لـ Small.
3. **Medium كان أقوى في أسئلة محددة:** غطى Q2 بمقدار 10/12 وQ3 وQ6 بالكامل. لكنه لم يسترجع أي دليل من Q1 وQ5 وQ8.
4. **Large كان قويًا في الأسئلة التي تجمّع دليلها داخل مقطع كبير:** أكمل Q1 وQ6 وQ7 وQ8، لكنه أخفق في Q2 وQ4 وQ5، وغطى Q3 جزئيًا.
5. **Small كان الأضعف إجمالًا:** 22/78 Anchor فقط، مع سؤال مكتمل واحد. صغر القطعة قد يفصل أجزاء الدليل أو يجعل كل جزء أقل قدرة على تمثيل سياقه.

### هل يوجد إعداد فائز؟

لو استخدمنا أرقام هذا الـ Benchmark فقط، فإن Large يتصدر بفارق ضئيل في التغطية الإجمالية وعدد الأسئلة المكتملة. لكن الاستنتاج الأدق هو أن **لا يوجد حجم Chunk واحد كان الأفضل لكل الأسئلة**؛ أداء كل إعداد اختلف باختلاف مكان الدليل وطبيعة السؤال وتوزيعه داخل المستند.

- Large مناسب في هذه العينة للأسئلة التي يستفيد استرجاعها من سياق أوسع.
- Medium قدم توازنًا قويًا، وكان الأفضل في استرجاع أدلة Q2 وQ3، وشارك Large في إكمال Q6 وQ7.
- Small لم يحقق نتائج منافسة في هذه المقارنة.

بالتالي، لا أتعامل مع فارق 1 Anchor بين Large وMedium على أنه قاعدة عامة. الاختيار النهائي في نظام إنتاجي يحتاج Benchmark أوسع بأسئلة ومستندات أكثر، وقد يستفيد من Chunking يحافظ على حدود الفقرات والعناوين بدل الاعتماد على عدد الحروف فقط.

### تفسير الفشل

- فشل Large في Q2 رغم نجاح Medium يوضح أن القطعة الأكبر ليست دائمًا أفضل؛ قد تجمع موضوعات متعددة فتضعف ملاءمة المقطع للسؤال.
- فشل Medium في Q8 مقابل نجاح Large الكامل يوضح أن بعض قوائم التسليمات قد تحتاج سياقًا أطول حتى تظهر عناصرها معًا.
- Q4 ظل ضعيفًا في الإعدادات الثلاثة؛ Medium استرجع 2/7 فقط، بينما لم يظهر الدليل المطلوب في Top 3 مع Small وLarge.
- نجاح الإعدادات الثلاثة في Q7 (4/4) يشير إلى أن الدليل في هذا السؤال كان قابلًا للاسترجاع عبر الأحجام الثلاثة.

### ما الذي تقيسه الأرقام وما الذي لا تقيسه؟

هذه التجربة تقيس **استرجاع Anchors مرجعية محددة داخل أعلى 3 نتائج**. لا تقيس دقة إجابة مولد لغوي، ولا صحة الاستشهادات في إجابة نهائية، ولا معدل الهلوسة، ولا جودة RAG من البداية للنهاية. كما أن التغطية تتأثر باختيار الـ Anchors والصفحات المرجعية؛ وأي خطأ في Ground Truth قد يؤثر في التقييم.

## 10. حدود التجربة

- المقارنة الرقمية هنا خاصة بـ Fixed-character chunking؛ طرق Paragraph-aware وSemantic اتشرحت للمقارنة المفاهيمية، لكنها مش داخلة كإعدادات تشغيل في الـ Benchmark.
- حجم الـ Chunk مقاس بالحروف، مش Tokens.
- مواضع الدليل المرجعية والـ Anchors متحددة يدويًا؛ فلازم مراجعتها مقابل صفحات المستند لو اتغير إصدار الـ PDF أو طريقة استخراج النص.
- النتيجة بتقيس ظهور الدليل في Top 3، مش جودة إجابة نموذج لغوي نهائية ولا صحة صياغته.
- النموذج المستخدم متعدد اللغات، لكن ده مش معناه إن أداءه مضمون على كل أنواع المستندات العربية.
- نتيجة التجربة تخص المستند والأسئلة والإعدادات دي؛ ومينفعش أعممها على كل أنظمة RAG أو كل قواعد البيانات.

---

## 11. الكود الكامل

الكود التالي هو جزء التنفيذ الخاص بالتجربة. حطيته في قسم مستقل عشان يفضل الشرح منفصل عن الكود، وفي نفس الوقت يبقى كل محتوى W7D2 موثق في ملف Markdown واحد.

> الكود بيقرأ ملف PDF من `data/source_document.pdf`، وبيستخدم ChromaDB وFastEmbed، وبيحفظ تفاصيل الـ 30 عملية استرجاع في `results/W7D2_chunking_comparison.csv`.

```python
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
SOURCE = "SGDS_SWARM_Engineering_Report_Civil_Edition(4) (1).pdf"
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 3

CONFIGS = [
    {"name": "small_180_30", "size": 180, "overlap": 30},
    {"name": "medium_700_120", "size": 700, "overlap": 120},
    {"name": "large_1600_250", "size": 1600, "overlap": 250},
]

QUESTIONS = [
    {"id": "Q1", "question": "ما الدور الذي تمثله كل طائرة داخل منظومة SGDS SWARM؟",
     "answer": "كل طائرة تمثل عقدة استشعار ومعالجة تنفذ الطيران وتستخلص المعلومات وتقيّم حالتها التشغيلية وتتبادل بيانات مختصرة وتتخذ قرارات محلية ضمن السياسات المحددة.",
     "evidence": [
         {"page": 3, "anchor": "عقدة استشعار ومعالجة"},
         {"page": 3, "anchor": "تنفيذ الطيران"},
         {"page": 3, "anchor": "استخلاص المعلومات"},
         {"page": 3, "anchor": "تقييم الحالة التشغيلية"},
         {"page": 3, "anchor": "تبادل بيانات مختصرة"},
         {"page": 3, "anchor": "اتخاذ قرارات محلية"},
     ]},
    {"id": "Q2", "question": "ما الطبقات الوظيفية العشر للمنظومة وما الطبقتان العرضيتان اللتان تغطيانها؟",
     "answer": "الطبقات الوظيفية العشر هي Airframe / Power / Mechanics وSensors & Payload وFlight Control وNavigation & Localization وCommunications وData Management وAI / Computer Vision وEvent & Decision Engine وFleet Management وUser Interface / Command Center. والطبقتان العرضيتان هما Cybersecurity وSafety / Reliability.",
     "evidence": [
         {"page": 8, "anchor": "Airframe / Power / Mechanics"},
         {"page": 8, "anchor": "Sensors & Payload"},
         {"page": 8, "anchor": "Flight Control"},
         {"page": 8, "anchor": "Navigation & Localization"},
         {"page": 8, "anchor": "Communications"},
         {"page": 8, "anchor": "Data Management"},
         {"page": 8, "anchor": "AI / Computer Vision"},
         {"page": 8, "anchor": "Event & Decision Engine"},
         {"page": 8, "anchor": "Fleet Management"},
         {"page": 8, "anchor": "User Interface / Command Center"},
         {"page": 9, "anchor": "Cybersecurity"},
         {"page": 9, "anchor": "Safety / Reliability"},
     ]},
    {"id": "Q3", "question": "لماذا لا يعتمد التقرير على عتبة بطارية ثابتة وما المدخلات المستخدمة لتقدير زمن المهمة الآمن؟",
     "answer": "لأن نسبة البطارية لا تمثل الاحتياطي نفسه في كل الظروف؛ ويتغير الاستهلاك حسب الحمولة والطقس والمسافة وحالة البطارية. ويعتمد التقدير على الطاقة المتبقية وطاقة العودة واحتياطي السلامة وتأثير الطقس وصحة البطارية واستهلاك الحمولة.",
     "evidence": [
         {"page": 13, "anchor": "لا تمثل مقدار الاحتياطي نفسه"},
         {"page": 13, "anchor": "عتبة ثابتة"},
         {"page": 13, "anchor": "الحمولة"},
         {"page": 13, "anchor": "الطقس"},
         {"page": 13, "anchor": "المسافة"},
         {"page": 13, "anchor": "حالة البطارية"},
         {"page": 13, "anchor": "Energy-Aware Mission Management"},
         {"page": 14, "anchor": "Remaining Energy"},
         {"page": 14, "anchor": "Return Energy"},
         {"page": 14, "anchor": "Safety Reserve"},
         {"page": 14, "anchor": "Weather Penalty"},
         {"page": 14, "anchor": "Battery Health"},
         {"page": 14, "anchor": "Payload Consumption"},
     ]},
    {"id": "Q4", "question": "ما المقاييس التشغيلية التي يوصي التقرير بتسجيلها بجانب mAP ولماذا؟",
     "answer": "يوصي التقرير بقياس الإنذارات الكاذبة ومعدل الإخفاق، وزمن الاستدلال ومعدل الإطارات واستهلاك الطاقة لكل استدلال ومعدل الاختناق الحراري؛ لأن mAP وحده لا يوضح الأداء التشغيلي وتكلفة تشغيل النموذج على العتاد.",
     "evidence": [
         {"page": 24, "anchor": "False Alerts/hour"},
         {"page": 24, "anchor": "Miss Rate"},
         {"page": 25, "anchor": "False Alarms per Operational Hour"},
         {"page": 25, "anchor": "Inference Latency"},
         {"page": 25, "anchor": "FPS"},
         {"page": 25, "anchor": "Watt/Inference"},
         {"page": 25, "anchor": "Thermal Throttling Rate"},
     ]},
    {"id": "Q5", "question": "كيف يتعامل النظام مع تراجع أداء إصدار نموذج جديد وكيف يراقب تغير البيانات؟",
     "answer": "يرصد Health Monitor تراجع الأداء ثم ينفذ Rollback للإصدار السابق. ويراقب Data Drift من خلال متوسط الثقة والاكتشافات غير المعروفة والإنذارات الكاذبة وتوزيع المشاهد وجودة الصور، ثم يطلب مراجعة النموذج عند اكتشاف الانحراف.",
     "evidence": [
         {"page": 26, "anchor": "Health monitor detects degradation"},
         {"page": 26, "anchor": "Rollback"},
         {"page": 26, "anchor": "Model v1.4 restored"},
         {"page": 27, "anchor": "Average confidence"},
         {"page": 27, "anchor": "Unknown detections"},
         {"page": 27, "anchor": "False alarms"},
         {"page": 27, "anchor": "Scene distribution"},
         {"page": 27, "anchor": "Image quality"},
         {"page": 27, "anchor": "Model Review Required"},
     ]},
    {"id": "Q6", "question": "ما الإجراءات المطلوبة لإدارة مفاتيح وحدات الأسطول وتأمين تحديثات البرامج؟",
     "answer": "يجب تحديد مسؤولية إنشاء المفاتيح وتوزيعها وتخزينها وصلاحيات الوصول إليها وتدويرها وإلغائها وإجراء التعامل مع فقدان الجهاز. والتحديث الآمن يمر بحزمة موقعة والتحقق من التوقيع والتوافق والتثبيت وفحص الصحة والاعتماد، مع الرجوع للإصدار السابق عند الفشل.",
     "evidence": [
         {"page": 32, "anchor": "إنشاء المفاتيح وتوزيعها"},
         {"page": 32, "anchor": "تخزين المفاتيح"},
         {"page": 32, "anchor": "صلاحيات الوصول"},
         {"page": 32, "anchor": "تدوير المفاتيح وإلغائها"},
         {"page": 32, "anchor": "فقدان طائرة"},
         {"page": 32, "anchor": "Certificate Revoked"},
         {"page": 32, "anchor": "Signed Package"},
         {"page": 32, "anchor": "Signature Verification"},
         {"page": 32, "anchor": "Compatibility Check"},
         {"page": 32, "anchor": "Install"},
         {"page": 32, "anchor": "Health Check"},
         {"page": 32, "anchor": "Commit"},
         {"page": 32, "anchor": "Rollback"},
     ]},
    {"id": "Q7", "question": "ما المبدأ الذي يجب أن يحكم تصرف الطائرة عند فقدان اتصال الشبكة؟",
     "answer": "فقدان الشبكة لا يعني فقدان الطائرة، ويجب أن تحتفظ كل طائرة بمنطق سلامة محلي وألا تنتظر أوامر الخادم عند انقطاع الاتصال.",
     "evidence": [
         {"page": 44, "anchor": "Loss of Network"},
         {"page": 44, "anchor": "Loss of Aircraft"},
         {"page": 44, "anchor": "Local Safety Logic"},
         {"page": 44, "anchor": "فقدان الاتصال"},
     ]},
    {"id": "Q8", "question": "ما التسليمات المطلوبة من فريق الذكاء الاصطناعي بخلاف ملف model.pt؟",
     "answer": "التسليمات الأربعة عشر هي Dataset specification وDataset version وLabel policy وTraining pipeline وEvaluation report وModel artifact وTensorRT artifact وBenchmark وInference service وEvent API وModel card وFailure cases وDrift monitor وRollback version.",
     "evidence": [
         {"page": 49, "anchor": "Dataset specification"},
         {"page": 49, "anchor": "Dataset version"},
         {"page": 49, "anchor": "Label policy"},
         {"page": 49, "anchor": "Training pipeline"},
         {"page": 49, "anchor": "Evaluation report"},
         {"page": 49, "anchor": "Model artifact"},
         {"page": 49, "anchor": "TensorRT artifact"},
         {"page": 49, "anchor": "Benchmark"},
         {"page": 49, "anchor": "Inference service"},
         {"page": 49, "anchor": "Event API"},
         {"page": 49, "anchor": "Model card"},
         {"page": 49, "anchor": "Failure cases"},
         {"page": 49, "anchor": "Drift monitor"},
         {"page": 49, "anchor": "Rollback version"},
     ]},
    {"id": "Q9", "question": "ما قيمة mAP التي حققها النموذج فعليًا في اختبار ميداني موثق على جهاز Jetson المستهدف؟",
     "answer": "لا يقدم المستند قيمة mAP موثقة لاختبار ميداني على جهاز Jetson المستهدف. يذكر مقاييس مطلوبة ومعايير قبول، وليس نتيجة اختبار فعلية.",
     "evidence": [], "absence_check": True},
    {"id": "Q10", "question": "كم رحلة اختبار ميداني مكتملة نُفذت فعليًا وما متوسط زمن الطيران المقاس بالدقائق؟",
     "answer": "لا يقدم المستند عدد رحلات اختبار ميداني مكتملة أو متوسط زمن طيران مقاسًا بالدقائق.",
     "evidence": [], "absence_check": True},
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

def validate_ground_truth(pages):
    page_text = {p["page"]: norm(p["text"]) for p in pages}
    missing = []
    for q in QUESTIONS:
        for item in q["evidence"]:
            expected_page = item["page"]
            anchor = item["anchor"]
            if expected_page not in page_text or norm(anchor) not in page_text[expected_page]:
                missing.append(f'{q["id"]}: page {expected_page}: {anchor}')
    if missing:
        details = "\n".join(f"  - {item}" for item in missing)
        raise ValueError("Ground Truth anchors not found in their expected PDF pages:\n" + details)
    total = sum(len(q["evidence"]) for q in QUESTIONS)
    print(f"[GROUND TRUTH] Verified {total} anchors across {sum(bool(q['evidence']) for q in QUESTIONS)} answerable questions.")

def index_chunks(collection, model, chunks):
    for start in range(0, len(chunks), 64):
        batch = chunks[start:start + 64]
        vectors = [v.tolist() for v in model.embed([x["text"] for x in batch])]
        collection.upsert(ids=[x["id"] for x in batch],
                          documents=[x["text"] for x in batch],
                          metadatas=[x["meta"] for x in batch], embeddings=vectors)

def retrieve(collection, model, q):
    vector = next(model.embed([q["question"]])).tolist()
    result = collection.query(query_embeddings=[vector], n_results=min(TOP_K, collection.count()),
                              include=["documents", "metadatas", "distances"])
    hits = []
    for rank, (cid, doc, meta, distance) in enumerate(
        zip(result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]), 1):
        hits.append({"rank": rank, "id": cid, "doc": doc, "meta": meta, "distance": float(distance)})

    metadata_ok = all(
        h["meta"].get("source_file") == SOURCE
        and isinstance(h["meta"].get("page_number"), int)
        and h["meta"]["page_number"] > 0
        for h in hits
    )

    evidence = []
    for item in q["evidence"]:
        expected = norm(item["anchor"])
        match = next(
            (hit for hit in hits
             if hit["meta"].get("page_number") == item["page"]
             and expected in norm(hit["doc"])),
            None
        )
        evidence.append({
            "anchor": item["anchor"],
            "page": item["page"],
            "found": match is not None,
            "rank": match["rank"] if match else None,
            "chunk_id": match["id"] if match else None,
        })

    if q.get("absence_check"):
        status, coverage = "ABSENCE_CHECK", None
    else:
        found = sum(item["found"] for item in evidence)
        coverage = found / len(evidence) if evidence else 0.0
        status = "YES" if found == len(evidence) else ("PARTIAL" if found else "NO")

    return {"hits": hits, "metadata_ok": metadata_ok, "evidence": evidence,
            "status": status, "coverage": coverage}


def main():
    OUT.mkdir(exist_ok=True)
    DB.mkdir(parents=True, exist_ok=True)
    pages, page_count = load_pages()
    validate_ground_truth(pages)
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
        total_expected = total_found = 0
        question_coverages, first_ranks = [], []

        for q in QUESTIONS:
            r = retrieve(col, model, q)
            if q["evidence"]:
                expected_count = len(r["evidence"])
                found_count = sum(item["found"] for item in r["evidence"])
                total_expected += expected_count
                total_found += found_count
                question_coverages.append(r["coverage"])

                if r["status"] == "YES":
                    complete += 1
                elif r["status"] == "PARTIAL":
                    partial += 1
                else:
                    failed += 1

                found_ranks = [item["rank"] for item in r["evidence"] if item["found"]]
                if found_ranks:
                    first_ranks.append(min(found_ranks))

            top_pages = " > ".join(f"p.{h['meta'].get('page_number', '?')}" for h in r["hits"])
            if r["evidence"]:
                order = "; ".join(
                    f'{item["anchor"]}→#{item["rank"]}' if item["found"]
                    else f'{item["anchor"]}→missing'
                    for item in r["evidence"]
                )
            else:
                order = "معلومة غير موثقة؛ مراجعة غيابها منفصلة"

            expected = "; ".join(f'p.{item["page"]}: {item["anchor"]}' for item in q["evidence"])
            if not expected:
                expected = "ABSENCE_CHECK"

            rows.append({
                "qid": q["id"], "question": q["question"], "config": cfg["name"],
                "size": cfg["size"], "overlap": cfg["overlap"],
                "expected": expected, "status": r["status"], "order": order,
                "top_pages": top_pages, "metadata": r["metadata_ok"],
                "hits": r["hits"], "evidence": r["evidence"],
            })
            print(f"  {q['id']} [{r['status']}] | top3={top_pages or 'no hits'} | metadata={'OK' if r['metadata_ok'] else 'ERROR'}"
                  + (f" | evidence={sum(x['found'] for x in r['evidence'])}/{len(r['evidence'])}"
                     if r["evidence"] else ""))

        overall_coverage = total_found / total_expected if total_expected else 0.0
        stats[cfg["name"]] = {
            "complete": int(complete), "partial": int(partial), "failed": int(failed),
            "coverage": overall_coverage,
            "mean_question_coverage": statistics.mean(question_coverages) if question_coverages else 0.0,
            "rank": statistics.mean(first_ranks) if first_ranks else 4.0,
            "evidence_found": total_found, "evidence_total": total_expected,
            "chunks": col.count(),
        }
        s = stats[cfg["name"]]
        print(f"[SUMMARY] complete={s['complete']}/8 partial={s['partial']}/8 failed={s['failed']}/8 "
              f"evidence={s['evidence_found']}/{s['evidence_total']} "
              f"coverage={s['coverage']:.1%} mean_question_coverage={s['mean_question_coverage']:.1%}")

    fields = ["question_id", "question", "configuration", "chunk_size_chars", "overlap_chars",
              "expected_evidence", "answer_in_top3", "evidence_rank_order", "evidence_details",
              "top3_pages", "metadata_verified",
              "top1_page", "top1_distance", "top1_source", "top1_chunk_id", "top1_preview",
              "top2_page", "top2_distance", "top2_source", "top2_chunk_id", "top2_preview",
              "top3_page", "top3_distance", "top3_source", "top3_chunk_id", "top3_preview"]
    with CSV_FILE.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            details = " | ".join(
                f'{item["anchor"]} [page={item["page"]}, found={item["found"]}, '
                f'rank={item["rank"]}, chunk={item["chunk_id"]}]'
                for item in row["evidence"]
            ) or "ABSENCE_CHECK: source absence is assessed separately"
            out = {
                "question_id": row["qid"], "question": row["question"],
                "configuration": row["config"], "chunk_size_chars": row["size"],
                "overlap_chars": row["overlap"], "expected_evidence": row["expected"],
                "answer_in_top3": row["status"], "evidence_rank_order": row["order"],
                "evidence_details": details, "top3_pages": row["top_pages"],
                "metadata_verified": row["metadata"],
            }
            for i in range(1, 4):
                h = row["hits"][i - 1] if len(row["hits"]) >= i else None
                out[f"top{i}_page"] = h["meta"].get("page_number", "") if h else ""
                out[f"top{i}_distance"] = f'{h["distance"]:.6f}' if h else ""
                out[f"top{i}_source"] = h["meta"].get("source_file", "") if h else ""
                out[f"top{i}_chunk_id"] = h["id"] if h else ""
                out[f"top{i}_preview"] = re.sub(r"\s+", " ", h["doc"])[:220] if h else ""
            writer.writerow(out)

    failures = [r for r in rows if r["status"] in {"NO", "PARTIAL"}]
    if failures:
        worst = min(failures, key=lambda r: (
            sum(item["found"] for item in r["evidence"]) / len(r["evidence"]),
            r["qid"]
        ))
        worst_text = (
            f'أسوأ حالة في نتائج المقارنة: {worst["qid"]} مع {worst["config"]}؛ '
            f'تغطية الأدلة {sum(x["found"] for x in worst["evidence"])}/{len(worst["evidence"])}. '
            f'الأدلة وترتيبها: {worst["order"]}'
        )
    else:
        worst_text = "كل الأسئلة القابلة للإجابة اكتملت في كل الإعدادات."

    winner_name, winner = max(
        stats.items(),
        key=lambda x: (x[1]["coverage"], x[1]["complete"], -x[1]["rank"])
    )
    meta_total = sum(len(r["hits"]) for r in rows)
    meta_valid = sum(
        1 for r in rows for h in r["hits"]
        if h["meta"].get("source_file") == SOURCE
        and isinstance(h["meta"].get("page_number"), int)
        and h["meta"]["page_number"] > 0
    )

    print("\n=== FINAL VERIFICATION ===")
    print(f"Comparison rows: {len(rows)} / 30")
    print(f"Valid source metadata: {meta_valid}/{meta_total}")
    print(f"Best configuration: {winner_name} ({winner['coverage']:.1%} evidence coverage)")
    print(f"Mean question coverage: {winner['mean_question_coverage']:.1%}")
    print(f"Worst case: {worst_text}")
    print("Q9/Q10: ABSENCE_CHECK — لا يتم اعتبار عدم الاسترجاع وحده إثباتًا لغياب المعلومة.")
    print(f"CSV: {CSV_FILE}")
    print("[DONE] راجعي الأدلة ومقتطفات Top 3 قبل اعتماد الاستنتاج النهائي.")

if __name__ == "__main__":
    main()
```

## 12. طريقة التشغيل

### تجهيز مكان ملف المصدر

من داخل مجلد المستودع، جهزي المجلدات:

```bash
mkdir -p data results
```

حطي التقرير الأصلي داخل مجلد `data` بالاسم ده بالضبط:

```text
data/source_document.pdf
```

### تثبيت المتطلبات

```bash
python -m pip install -r requirements.txt
```

### تشغيل التجربة

بعد استخراج الكود من قسم **الكود الكامل** وحفظه مؤقتًا باسم `W7D2_Chunking_Experiment.py` داخل مجلد المستودع، شغليه بالأمر:

```bash
python W7D2_Chunking_Experiment.py
```

### الملفات الناتجة من التشغيل

- `results/W7D2_chunking_comparison.csv`: تفاصيل الأسئلة العشرة مع كل إعداد، ونتائج Top 3، وترتيب الدليل، والـ Metadata، ومقتطفات النصوص المسترجعة.
- `chroma_db/w7d2_chunking/`: قاعدة بيانات Chroma المحلية الناتجة عن التجربة.

**تنبيه توثيقي:** ملف Markdown ده هو ملف التسليم والشرح الوحيد الخاص بـ W7D2. ملف Python مش هيفضل ملفًا مستقلًا في جذر المستودع؛ الكود موجود داخل القسم المخصص له هنا. ملفات النتائج وقاعدة البيانات بتتولد محليًا وقت التنفيذ، ونتائج التنفيذ موثقة في القسمين 8 و9، ولا تعني ترشيح إعداد واحد على أنه الأفضل لكل أنواع المستندات.
