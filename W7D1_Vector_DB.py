from __future__ import annotations
import csv, re, unicodedata
from pathlib import Path
import chromadb
from fastembed import TextEmbedding
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parent
PDF=ROOT/"data"/"source_document.pdf"
DB=ROOT/"chroma_db"
RESULTS=ROOT/"results"
MODEL="intfloat/multilingual-e5-small"
SOURCE="SGDS_SWARM_Engineering_Report_Civil_Edition.pdf"
CONFIGS=[
 {"name":"small_180_30","size":180,"overlap":30},
 {"name":"medium_700_120","size":700,"overlap":120},
 {"name":"large_1600_250","size":1600,"overlap":250},
]
QUESTIONS=[
 {"id":"Q1","q":"ما الفكرة الأساسية لمنظومة SGDS SWARM، وما الفرق بين دور الطائرات ودور محطة التحكم الأرضية؟","pages":[3,4],"answer":"منظومة مراقبة جوية مدنية موزعة؛ الطائرات عقد استشعار ومعالجة محلية وتتعاون في تنفيذ المهام، ومحطة التحكم تراقب الأسطول وتدير المهام والأحداث مع إشراف بشري على القرارات الحساسة."},
 {"id":"Q2","q":"ما الطبقات الوظيفية العشر المقترحة للمنظومة، وما الطبقتان اللتان تغطيان النظام كله؟","pages":[8,9],"answer":"الطبقات: Airframe/Power/Mechanics، Sensors/Payload، Flight Control، Navigation/Localization، Communications، Data Management، AI/Computer Vision، Event/Decision Engine، Fleet Management، User Interface/Command Center. والطبقتان العرضيتان Cybersecurity وSafety/Reliability."},
 {"id":"Q3","q":"لماذا يرفض التقرير الاعتماد على عتبة بطارية ثابتة 20%، وما المدخلات التي تدخل في تقدير الطاقة الآمنة؟","pages":[13,14],"answer":"لأن الاستهلاك يتغير مع الحمولة والطقس والمسافة وصحة البطارية والحرارة؛ ويُحسب القرار من الطاقة المتبقية وطاقة العودة واحتياطي السلامة وعقوبة الطقس وصحة البطارية واستهلاك الحمولة."},
 {"id":"Q4","q":"ما المقاييس التي يوصي التقرير بتوثيقها لتقييم نموذج الكشف، ولماذا لا تكفي Accuracy وحدها؟","pages":[24,25],"answer":"Precision وRecall وmAP وFalse Alerts/hour وMiss Rate وInference Latency وFPS وWatt/Inference وThermal Throttling Rate؛ لأن Accuracy وحدها لا تصف الإنذارات الكاذبة والأحداث الفائتة وكلفة التشغيل."},
 {"id":"Q5","q":"كيف يتعامل النظام مع تراجع أداء إصدار نموذج جديد، وكيف يراقب تغير البيانات بمرور الوقت؟","pages":[26,27],"answer":"Health Monitor يرصد التدهور ثم Rollback للإصدار السابق؛ ويراقب Data Drift عبر Average confidence وUnknown detections وFalse alarms وScene distribution وImage quality، ثم يطلب Model Review."},
 {"id":"Q6","q":"ما الإجراءات التي يجب أن تشملها إدارة المفاتيح والتحديثات الآمنة، ولماذا لا يكفي ذكر AES-256؟","pages":[7,32,47],"answer":"تحديد إنشاء المفاتيح وتوزيعها وتخزينها وصلاحيات الوصول وتدويرها وإلغائها، مع تحديثات موقعة والتحقق من التوقيع والتوافق وفحص الصحة وRollback؛ فالأمن يشمل التوثيق وحماية النقل وإدارة المفاتيح والتدقيق."},
 {"id":"Q7","q":"ما المبدأ الذي يجب أن يحكم تصرف الطائرة عند فقدان اتصال الشبكة؟","pages":[44],"answer":"Loss of Network لا يعني Loss of Aircraft؛ يجب أن تحتفظ كل طائرة بمنطق سلامة محلي وألا تنتظر أوامر الخادم عند انقطاع الاتصال."},
 {"id":"Q8","q":"ما التسليمات المطلوبة من فريق الذكاء الاصطناعي بخلاف ملف model.pt؟","pages":[49],"answer":"Dataset specification/version، Label policy، Training pipeline، Evaluation report، Model artifact، TensorRT artifact، Benchmark، Inference service، Event API، Model card، Failure cases، Drift monitor، Rollback version."},
 {"id":"Q9","q":"ما قيمة mAP الفعلية التي حققها نموذج الكشف في اختبار ميداني مكتمل على Jetson المستهدف؟","pages":[],"answer":"غير موجودة؛ المستند يحدد مقاييس يجب قياسها ولا يقدم نتيجة mAP ميدانية موثقة."},
 {"id":"Q10","q":"كم رحلة اختبار ميداني مكتملة نُفذت فعليًا، وما متوسط زمن الطيران المقاس بالدقائق؟","pages":[],"answer":"غير موجودة؛ لا توجد حصيلة موثقة لعدد الرحلات أو متوسط زمن الطيران المقاس."},
]

def clean(s):
 s=unicodedata.normalize("NFC",s or "")
 s=re.sub(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff]","",s)
 s=s.replace("\x00"," ")
 return re.sub(r"\n{3,}","\n\n",re.sub(r"[ \t]+"," ",s)).strip()

def load_pages():
 if not PDF.exists(): raise FileNotFoundError(f"ضعي ملف PDF داخل {PDF.parent}")
 reader=PdfReader(str(PDF))
 if len(reader.pages)<5: raise ValueError("المستند أقل من 5 صفحات")
 pages=[{"page":i,"text":clean(p.extract_text() or "")} for i,p in enumerate(reader.pages,1)]
 pages=[p for p in pages if p["text"]]
 if len(pages)<5: raise ValueError("النص المستخرج أقل من 5 صفحات؛ قد يكون الملف صورًا ممسوحة")
 return pages,len(reader.pages)

def make_chunks(pages,cfg):
 out=[]; step=cfg["size"]-cfg["overlap"]
 for p in pages:
  start=i=0
  while start<len(p["text"]):
   part=p["text"][start:start+cfg["size"]].strip()
   if part:
    out.append({"id":f"{cfg['name']}_p{p['page']}_c{i}","text":part,
     "meta":{"source_file":SOURCE,"page_number":p["page"],"chunk_index":i,
             "chunk_size":cfg["size"],"overlap":cfg["overlap"],"strategy":"fixed_character"}})
    i+=1
   if start+cfg["size"]>=len(p["text"]): break
   start+=step
 return out

def store(model,chunks,col):
 for start in range(0,len(chunks),64):
  batch=chunks[start:start+64]
  vec=list(model.embed(["passage: "+x["text"] for x in batch]))
  col.upsert(ids=[x["id"] for x in batch],documents=[x["text"] for x in batch],
   metadatas=[x["meta"] for x in batch],embeddings=[v.tolist() for v in vec])

def evaluate(res,expected):
 hits=[]; ok=True
 for rank,(doc,meta,dist) in enumerate(zip(res["documents"][0],res["metadatas"][0],res["distances"][0]),1):
  page=int(meta.get("page_number",-1)); ok=ok and bool(meta.get("source_file")) and page>0
  hits.append({"rank":rank,"page":page,"distance":float(dist),"preview":re.sub(r"\s+"," ",doc)[:160]})
 if not expected: return {"status":"N/A — unsupported","rank":"","order":"N/A","pages":[h["page"] for h in hits],"ok":ok,"hits":hits}
 rankmap={}
 for h in hits: rankmap.setdefault(h["page"],h["rank"])
 found=[p for p in expected if p in rankmap]
 status="YES" if len(found)==len(expected) else ("PARTIAL" if found else "NO")
 order="; ".join(f"p.{p}→{rankmap[p]}" if p in rankmap else f"p.{p}→missing" for p in expected)
 return {"status":status,"rank":min((rankmap[p] for p in found),default=""),"order":order,
         "pages":[h["page"] for h in hits],"ok":ok,"hits":hits}

def write_outputs(rows,stats,worst):
 RESULTS.mkdir(exist_ok=True)
 fields=["question_id","question","config","chunk_size","overlap","expected_pages","answer_in_top3",
         "first_relevant_rank","expected_page_rank_order","returned_pages","metadata_verified",
         "top1_page","top1_distance","top2_page","top2_distance","top3_page","top3_distance"]
 with (RESULTS/"comparison_results.csv").open("w",newline="",encoding="utf-8-sig") as f:
  w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
  for r in rows: w.writerow({k:r.get(k,"") for k in fields})
 lines=["# W7D1 — Arabic Vector DB Chunking Experiment","","## Source & method",
  f"- PDF: {SOURCE} (53 pages; verify count in execution output).","- Embeddings: "+MODEL,
  "- ChromaDB with cosine distance; Top 3 retrieval.","",
  "## Questions and reference answers",""]
 for q in QUESTIONS:
  lines += [f"### {q['id']} — {q['q']}",f"- Expected evidence pages: {q['pages'] or 'not present'}",f"- Reference answer: {q['answer']}",""]
 lines += ["## Configurations","","| Config | Size | Overlap |","|---|---:|---:|",
  "| small_180_30 | 180 | 30 |","| medium_700_120 | 700 | 120 |","| large_1600_250 | 1600 | 250 |","",
  "## Results: 10 questions × 3 configurations","","| Q | Config | Expected pages | Top-3 pages | Answer in Top 3? | Evidence rank | Metadata |",
  "|---|---|---|---|---|---|---|"]
 for r in rows:
  lines.append(f"| {r['question_id']} | {r['config']} | {r['expected_pages']} | {r['returned_pages']} | {r['answer_in_top3']} | {r['expected_page_rank_order']} | {'OK' if r['metadata_verified'] else 'ERROR'} |")
 lines += ["","## Summary",""]
 for c in CONFIGS:
  s=stats[c["name"]]
  lines.append(f"- **{c['name']}**: complete {s['complete']}/8; partial {s['partial']}/8; failed {s['failed']}/8; average first-hit rank {s['avg']}.")
 best_name,best=sorted(stats.items(),key=lambda x:(x[1]["complete"],x[1]["partial"],-x[1]["failed"],-x[1]["avg_num"]),reverse=True)[0]
 lines += ["","## Analysis","",
  f"Best setting by complete evidence coverage in Top 3: **{best_name}** ({best['complete']}/8 complete). This result applies only to this source, query set and embedding model.",
  f"**Worst observed failure:** {worst}","",
  "Q9 and Q10 are intentionally unanswerable from the source. Vector search still returns nearest chunks; nearest-neighbor rank alone is not proof that an answer exists. A production RAG pipeline needs an independently calibrated abstention/relevance policy.",
  "","## Concepts","",
  "- Chunk: a document segment embedded independently; embedding an entire long document can mix topics and hide local evidence.",
  "- Too-small chunks can split a sentence or separate a claim from its explanation; too-large chunks can add unrelated content and retrieval noise.",
  "- Overlap preserves context across boundaries; excessive overlap creates duplicate chunks and wastes storage.",
  "- Fixed-character splitting is simple but can cut sentences/paragraphs. Paragraph/sentence splitting preserves natural boundaries but creates variable sizes. Semantic splitting uses meaning shifts and is more complex and model-dependent.",
  "- Each chunk stores source_file, page_number, chunk_index, chunk_size and overlap. The benchmark checks source metadata on every returned result.",
  "","## Scope limitation","",
  "The numeric comparison uses fixed-character chunking only, so size and overlap are the variables. Paragraph/sentence and semantic chunking are explained but not included in the 30-run benchmark.",
  "","## Run","",
  "~~~bash","python -m pip install -r requirements.txt","python W7D1_Vector_DB.py","~~~","",
  "The script writes results/comparison_results.csv and replaces this README with the measured results."]
 (ROOT/"README.md").write_text("\n".join(lines),encoding="utf-8")

def main():
 pages,total=load_pages()
 print(f"[OK] PDF pages: {total}; pages with extractable text: {len(pages)}")
 print(f"[INFO] Loading multilingual model: {MODEL} (first run downloads model files)")
 model=TextEmbedding(model_name=MODEL)
 client=chromadb.PersistentClient(path=str(DB))
 rows=[]; stats={}
 for cfg in CONFIGS:
  name=cfg["name"]; chunks=make_chunks(pages,cfg)
  try: client.delete_collection(name=name)
  except Exception: pass
  col=client.create_collection(name=name,metadata={"hnsw:space":"cosine"})
  store(model,chunks,col)
  print(f"\nCONFIG {name}: size={cfg['size']} overlap={cfg['overlap']} chunks={col.count()}")
  complete=partial=failed=0; ranks=[]
  for q in QUESTIONS:
   qv=next(model.embed(["query: "+q["q"]])).tolist()
   res=col.query(query_embeddings=[qv],n_results=min(3,col.count()),include=["documents","metadatas","distances"])
   ev=evaluate(res,q["pages"])
   if q["pages"]:
    if ev["status"]=="YES": complete+=1
    elif ev["status"]=="PARTIAL": partial+=1
    else: failed+=1
    if ev["rank"]!="": ranks.append(int(ev["rank"]))
   print(f"{q['id']} [{ev['status']}] expected={q['pages'] or 'ABSENT'} | "+
         " | ".join(f"#{h['rank']} p.{h['page']} d={h['distance']:.4f}" for h in ev["hits"]))
   row={"question_id":q["id"],"question":q["q"],"config":name,"chunk_size":cfg["size"],
    "overlap":cfg["overlap"],"expected_pages":";".join(map(str,q["pages"])) or "ABSENT",
    "answer_in_top3":ev["status"],"first_relevant_rank":ev["rank"],
    "expected_page_rank_order":ev["order"],"returned_pages":";".join(map(str,ev["pages"])),
    "metadata_verified":ev["ok"]}
   for i in range(3):
    h=ev["hits"][i] if i<len(ev["hits"]) else {}
    row[f"top{i+1}_page"]=h.get("page",""); row[f"top{i+1}_distance"]=h.get("distance","")
   rows.append(row)
  avg=sum(ranks)/len(ranks) if ranks else 0
  stats[name]={"complete":complete,"partial":partial,"failed":failed,"avg":f"{avg:.2f}","avg_num":avg}
  print(f"[SUMMARY] complete={complete}/8 partial={partial}/8 failed={failed}/8 average rank={avg:.2f}")
 failures=[r for r in rows if r["expected_pages"]!="ABSENT" and r["answer_in_top3"]!="YES"]
 if failures:
  w=min(failures,key=lambda r:(sum(p in r["expected_pages"].split(";") for p in r["returned_pages"].split(";")),-int(r["first_relevant_rank"] or 99)))
  worst=f"{w['question_id']} / {w['config']}: expected pages {w['expected_pages']}; returned {w['returned_pages']}; status {w['answer_in_top3']}."
 else: worst="No complete failure; inspect any PARTIAL rows."
 write_outputs(rows,stats,worst)
 print("\n=== FINAL VERIFICATION ===")
 print(f"Rows written: {len(rows)} (expected 30)")
 print(f"Metadata verified: {sum(r['metadata_verified'] for r in rows)}/30")
 print(f"CSV: {RESULTS/'comparison_results.csv'}")
 print("README.md updated with the measured comparison.")
 print("[DONE] Review README.md before submission.")

if __name__=="__main__": main()
