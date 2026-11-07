# PHANTOMS AI — Week 7 / Day 2
## تجربة Chunking وتأثيره على جودة Retrieval في RAG

> **نوع المهمة:** تجربة عملية ومقارنة استرجاع  
> **المستند المستخدم:** `SGDS_SWARM_Engineering_Report_Civil_Edition.pdf`  
> **عدد صفحات المستند المذكور في بيانات المشروع:** 53 صفحة  
> **قاعدة البيانات:** ChromaDB  
> **نموذج التضمين:** `intfloat/multilingual-e5-small`  
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

اخترت تقرير `SGDS_SWARM_Engineering_Report_Civil_Edition.pdf` لأنه مستند هندسي حقيقي ومتعدد الأقسام، وفيه معلومات عن معمارية النظام، والعتاد، والذكاء الاصطناعي، وإدارة البيانات، والاتصالات، والأمن، والسلامة، وخطة التنفيذ.

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
| Q4 | إيه المقاييس التشغيلية اللي التقرير بيوصي بتسجيلها بجانب mAP وليه؟ | بيوصي بتسجيل False Alarms per Operational Hour وInference Latency وFPS وWatt/Inference وThermal Throttling Rate؛ لأن mAP لوحده مش بيكشف الإنذارات الكاذبة أو تكلفة وسرعة التشغيل على العتاد. | 25 |
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
| `source_file` | `SGDS_SWARM_Engineering_Report_Civil_Edition.pdf` | اسم ملف المصدر |
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
5. إنشاء Embeddings للنصوص باستخدام `intfloat/multilingual-e5-small`.
6. تخزين الـ Embeddings والنصوص والـ Metadata في ChromaDB باستخدام Cosine distance.
7. تحويل كل سؤال إلى Embedding واسترجاع أعلى 3 نتائج.
8. مقارنة النتائج بمواضع الدليل المرجعية المحددة لكل سؤال.
9. تسجيل ترتيب ظهور الدليل، وصفحات النتائج الثلاث، وقيم المسافة، ومعرفات الـ Chunks، والتحقق من الـ Metadata.
10. حفظ الـ 30 تجربة في ملف CSV مساعد للتحليل والمراجعة.

## 8. جدول النتائج

**حالة التنفيذ الحالية:** جدول القياسات الفعلية لسه محتاج يتملأ بعد تشغيل السكربت على ملف الـ PDF في البيئة المحلية. مش هحط أرقام تقديرية أو أنقل نتائج قديمة اتقاست بطريقة مختلفة؛ لأن المطلوب هنا مقارنة باستخدام ChromaDB وDense Embeddings.

الرموز اللي هستخدمها بعد التشغيل:

- **YES:** كل مجموعات الدليل المطلوبة للسؤال ظهرت في Top 3.
- **PARTIAL:** جزء من الدليل ظهر، لكن مش كل الأدلة المطلوبة.
- **NO:** ولا مجموعة دليل مطلوبة ظهرت في Top 3.
- **NOT_IN_SOURCE:** السؤال إجابته غير موجودة في المستند أصلًا.
- **Metadata OK:** كل نتيجة مسترجعة فيها اسم الملف ورقم صفحة صحيحين.

| السؤال | Small 180/30 | Medium 700/120 | Large 1600/250 |
|---|---|---|---|
| Q1 | بانتظار التشغيل | بانتظار التشغيل | بانتظار التشغيل |
| Q2 | بانتظار التشغيل | بانتظار التشغيل | بانتظار التشغيل |
| Q3 | بانتظار التشغيل | بانتظار التشغيل | بانتظار التشغيل |
| Q4 | بانتظار التشغيل | بانتظار التشغيل | بانتظار التشغيل |
| Q5 | بانتظار التشغيل | بانتظار التشغيل | بانتظار التشغيل |
| Q6 | بانتظار التشغيل | بانتظار التشغيل | بانتظار التشغيل |
| Q7 | بانتظار التشغيل | بانتظار التشغيل | بانتظار التشغيل |
| Q8 | بانتظار التشغيل | بانتظار التشغيل | بانتظار التشغيل |
| Q9 | NOT_IN_SOURCE | NOT_IN_SOURCE | NOT_IN_SOURCE |
| Q10 | NOT_IN_SOURCE | NOT_IN_SOURCE | NOT_IN_SOURCE |

بعد التشغيل، هراجع لكل خلية: هل الدليل ظهر؟ ظهر في أنهي ترتيب؟ وإيه صفحات الـ Top 3؟ وهل الـ Metadata سليمة؟ تفاصيل النتائج ومقتطفات النصوص والمسافات هتكون في ملف CSV اللي السكربت بيطلعه للمراجعة.

## 9. تحليل النتائج والاستنتاج

مش هحدد الإعداد الأفضل قبل ما أشغّل الـ 30 عملية Retrieval وأراجع النتائج. معيار الاختيار الأساسي عندي هو اكتمال الدليل في Top 3، وبعده نسبة تغطية الأدلة، وبعدها ترتيب أول دليل ظهر.

### إزاي هحدد أفضل إعداد؟

هقارن بين:

- عدد الأسئلة المكتملة من أصل 8 أسئلة قابلة للإجابة.
- عدد الأسئلة اللي ظهر فيها جزء من الدليل بس.
- عدد الأسئلة اللي فشل الاسترجاع في إظهار دليلها.
- متوسط تغطية مجموعات الأدلة المطلوبة.
- متوسط ترتيب أول دليل ظهر.
- سلامة الـ Metadata في كل النتائج.

### إيه اللي هاعتبره أسوأ فشل؟

أسوأ فشل مش بالضرورة السؤال اللي مفيش له نتيجة قريبة؛ ممكن يكون السؤال اللي إجابته موزعة على أكتر من موضع، والنظام يرجّع جزء واحد بس ويخلّي الجزء التاني بره Top 3. ده ممكن يخلي نموذج RAG يكوّن إجابة ناقصة أو يربط معلومات من غير دليل كافي.

### ليه Q9 وQ10 مهمين؟

لأن ChromaDB هترجع أقرب نتائج حتى لو مفيش إجابة فعلية في المستند. وجود نتائج متشابهة في المعنى مش معناه إن المصدر جاوب على السؤال.

بالنسبة للسؤالين دول، السلوك الصحيح من نظام RAG هو إنه يوضح إن المعلومة مش موجودة في المصدر، بدل ما يخمّن قيمة mAP أو عدد رحلات أو متوسط زمن طيران.

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

    print("\n=== FINAL VERIFICATION ===")
    print(f"Comparison rows: {len(rows)} / 30")
    print(f"Valid source metadata: {meta_valid}/{meta_total}")
    print(f"Best configuration: {winner_name} ({winner['coverage']:.1%} evidence coverage)")
    print(f"CSV: {CSV_FILE}")
    print("[DONE] راجعي أسوأ حالة ومقتطفات النتائج قبل التسليم.")

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

**تنبيه توثيقي:** ملف Markdown ده هو ملف التسليم والشرح الوحيد الخاص بـ W7D2. ملف Python مش هيفضل ملفًا مستقلًا في جذر المستودع؛ الكود موجود داخل القسم المخصص له هنا. ملفات النتائج وقاعدة البيانات بتتولد محليًا وقت التنفيذ، ومش هاعتبر التجربة مكتملة أو أختار إعدادًا فائزًا قبل مراجعة النتائج الفعلية.
