from __future__ import annotations
import argparse, csv, json, os, re, sys, time
from pathlib import Path
import chromadb, requests
from fastembed import TextEmbedding
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
PDF=PROJECT/"data"/"source_document.pdf"
BASE_DB=PROJECT/"chroma_db"/"w7d2_chunking"
BASE_COLLECTION="large_1600_250"
SOURCE="SGDS_SWARM_Engineering_Report_Civil_Edition(4) (1).pdf"
BASE_MODEL="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
WRONG_MODEL="sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_URL=os.getenv("OLLAMA_URL","http://localhost:11434/api/chat")
OLLAMA_MODEL=os.getenv("OLLAMA_MODEL","qwen2.5:3b")
TIMEOUT=int(os.getenv("OLLAMA_TIMEOUT","600"))
OUT=ROOT/"results"
INDEXES=ROOT/"indexes"
ABSTAIN="مش موجود في المستند"
QUESTIONS=[
("Q1","ما الدور الذي تمثله كل طائرة داخل منظومة SGDS SWARM؟"),
("Q2","ما الطبقات الوظيفية العشر للمنظومة وما الطبقتان العرضيتان اللتان تغطيانها؟"),
("Q3","لماذا لا يعتمد التقرير على عتبة بطارية ثابتة وما المدخلات المستخدمة لتقدير زمن المهمة الآمن؟"),
("Q4","ما المقاييس التشغيلية التي يوصي التقرير بتسجيلها بجانب mAP ولماذا؟"),
("Q5","كيف يتعامل النظام مع تراجع أداء إصدار نموذج جديد وكيف يراقب تغير البيانات؟"),
("Q6","ما الإجراءات المطلوبة لإدارة مفاتيح وحدات الأسطول وتأمين تحديثات البرامج؟"),
("Q7","ما المبدأ الذي يجب أن يحكم تصرف الطائرة عند فقدان اتصال الشبكة؟"),
("Q8","ما التسليمات المطلوبة من فريق الذكاء الاصطناعي بخلاف ملف model.pt؟"),
("Q9","ما قيمة mAP التي حققها النموذج فعليًا في اختبار ميداني موثق على جهاز Jetson المستهدف؟"),
("Q10","كم رحلة اختبار ميداني مكتملة نُفذت فعليًا وما متوسط زمن الطيران المقاس بالدقائق؟"),
]
SYSTEM=f"""أنت مساعد يجيب اعتمادًا على سياق المستند المرفق فقط.
استخدم المعلومات الموجودة في Context فقط، وممنوع استخدام معلوماتك العامة.
لا تكمل أي معلومة ناقصة بالتخمين.
إذا كان الدليل غير موجود بوضوح أو غير كافٍ، أجب بالنص التالي حرفيًا:
{ABSTAIN}
لا تخترع أرقامًا أو أسماء أو نتائج أو أرقام صفحات.
اكتب إجابة مباشرة ومختصرة."""

DESCRIPTIONS={
"embedding":"تغيير نموذج الـ Embedding وقت السؤال فقط مع إبقاء فهرس W7D2 كما هو",
"chunk-small":"إعادة فهرسة التقرير بأجزاء صغيرة جدًا 80/10 بدل 1600/250",
"topk-1":"تقليل Top-k إلى 1",
"topk-high":"رفع Top-k إلى 1000",
"prompt":"إزالة قيود Grounding والسماح بالمعرفة العامة",
"metadata":"تغيير أسماء حقول المصدر والصفحة داخل Metadata",
}

def base_collection():
    if not BASE_DB.is_dir(): raise FileNotFoundError(f"قاعدة W7D2 غير موجودة: {BASE_DB}")
    client=chromadb.PersistentClient(path=str(BASE_DB))
    col=client.get_collection(BASE_COLLECTION)
    if not col.count(): raise RuntimeError("Collection فارغة")
    print(f"[BASE] collection={BASE_COLLECTION} chunks={col.count()}")
    return col

def chunk_index(size=80,overlap=10):
    if not PDF.is_file(): raise FileNotFoundError(f"ملف PDF غير موجود: {PDF}")
    name=f"fault_chunks_{size}_{overlap}"
    INDEXES.mkdir(parents=True,exist_ok=True)
    client=chromadb.PersistentClient(path=str(INDEXES/name))
    try:
        col=client.get_collection(name)
        if col.count(): print(f"[INDEX CACHE] {col.count()} chunks"); return col
    except Exception: pass
    try: client.delete_collection(name)
    except Exception: pass
    col=client.create_collection(name=name,metadata={"hnsw:space":"cosine"})
    reader=PdfReader(str(PDF)); emb=TextEmbedding(model_name=BASE_MODEL)
    docs=[]; metas=[]; ids=[]
    for pno,page in enumerate(reader.pages,1):
        body=re.sub(r"\s+"," ",page.extract_text() or "").strip()
        start=0; idx=0
        while start<len(body):
            part=body[start:start+size].strip()
            if part:
                docs.append(part); ids.append(f"p{pno}_c{idx}")
                metas.append({"source_file":SOURCE,"page_number":pno,"chunk_index":idx,
                              "chunk_size_chars":size,"overlap_chars":overlap,
                              "chunking_method":"debug_fixed_character_per_page"})
                idx+=1
            if start+size>=len(body): break
            start+=max(1,size-overlap)
    print(f"[REINDEX] pages={len(reader.pages)} chunks={len(docs)} size={size} overlap={overlap}")
    for start in range(0,len(docs),64):
        d=docs[start:start+64]
        col.add(ids=ids[start:start+64],documents=d,metadatas=metas[start:start+64],
                embeddings=[v.tolist() for v in emb.embed(d)])
        print(f"[REINDEX] {min(start+len(d),len(docs))}/{len(docs)}")
    return col

def broken_metadata_index(base):
    name="fault_metadata_renamed"; INDEXES.mkdir(parents=True,exist_ok=True)
    client=chromadb.PersistentClient(path=str(INDEXES/name))
    try: client.delete_collection(name)
    except Exception: pass
    target=client.create_collection(name=name,metadata={"hnsw:space":"cosine"})
    total=base.count()
    for off in range(0,total,100):
        b=base.get(limit=100,offset=off,include=["documents","metadatas","embeddings"])
        metas=[]
        for old in b["metadatas"]:
            m=dict(old or {})
            if "source_file" in m: m["filename"]=m.pop("source_file")
            if "page_number" in m: m["page"]=m.pop("page_number")
            metas.append(m)
        target.add(ids=b["ids"],documents=b["documents"],metadatas=metas,embeddings=b["embeddings"])
    print(f"[METADATA FAULT] cloned={target.count()} sample={target.get(limit=1,include=['metadatas'])['metadatas']}")
    return target

def ask(messages):
    r=requests.post(OLLAMA_URL,json={"model":OLLAMA_MODEL,"messages":messages,"stream":False,
        "keep_alive":"10m","options":{"temperature":0,"num_predict":450,"num_ctx":4096}},timeout=TIMEOUT)
    r.raise_for_status()
    answer=r.json().get("message",{}).get("content","").strip()
    if not answer: raise RuntimeError("Ollama returned an empty response")
    return answer

def run_question(col,emb,fault,qid,question):
    topk=1 if fault=="topk-1" else (1000 if fault=="topk-high" else 3)
    vec=next(emb.embed([question])).tolist()
    found=col.query(query_embeddings=[vec],n_results=min(topk,col.count()),
                    include=["documents","metadatas","distances"])
    docs=(found.get("documents") or [[]])[0]
    metas=(found.get("metadatas") or [[]])[0]
    dists=(found.get("distances") or [[]])[0]
    ctx=[]; hits=[]
    for rank,(doc,meta,dist) in enumerate(zip(docs,metas,dists),1):
        m=meta or {}; file=m.get("source_file","MISSING_SOURCE_FILE"); page=m.get("page_number","MISSING_PAGE_NUMBER")
        ctx.append(f"[الملف: {file} | الصفحة: {page}]\n{doc or ''}")
        hits.append({"rank":rank,"page":page,"source_file":file,"distance":round(float(dist),6),
                     "preview":re.sub(r"\s+"," ",doc or "")[:350],"metadata_keys":sorted(m.keys())})
    if fault=="prompt":
        system="أنت مساعد ذكي. أجب بأفضل ما تعرفه ويمكنك استخدام معرفتك العامة والاستنتاج حتى لو لم يظهر الدليل كاملًا في السياق."
        user=f"السؤال: {question}\n\nسياق قد يكون مفيدًا:\n{chr(10).join(ctx)}\n\nأجب بإجابة مفيدة."
    else:
        system=SYSTEM; user=f"السؤال: {question}\n\nContext:\n{chr(10).join(ctx)}\n\nأجب من السياق فقط."
    answer=ask([{"role":"system","content":system},{"role":"user","content":user}])
    return {"question":question,"answer":answer,
      "status":"ABSTAINED" if ABSTAIN in answer else "GENERATED_REQUIRES_REVIEW",
      "retrieved_count":len(hits),"retrieved":hits,
      "metadata_integrity":all(h["page"]!="MISSING_PAGE_NUMBER" and h["source_file"]!="MISSING_SOURCE_FILE" for h in hits),
      "error":""}

def run(fault):
    print("\n"+"="*76+f"\n[FAULT] {fault}: {DESCRIPTIONS[fault]}")
    print("[ISOLATION] قاعدة W7D2 الأصلية للقراءة فقط ولن يتم تعديلها")
    base=base_collection()
    if fault=="chunk-small": col=chunk_index(); model=BASE_MODEL
    elif fault=="metadata": col=broken_metadata_index(base); model=BASE_MODEL
    else: col=base; model=WRONG_MODEL if fault=="embedding" else BASE_MODEL
    print(f"[EMBEDDING] query_model={model}")
    emb=TextEmbedding(model_name=model)
    data={"fault":fault,"description":DESCRIPTIONS[fault],
          "settings":{"collection":col.name,"collection_count":col.count(),"query_embedding_model":model,
          "indexed_embedding_model":BASE_MODEL,"top_k":1 if fault=="topk-1" else 1000 if fault=="topk-high" else 3,
          "prompt":"ungrounded" if fault=="prompt" else "grounded","llm":OLLAMA_MODEL},"questions":{}}
    OUT.mkdir(parents=True,exist_ok=True); jp=OUT/f"{fault}.json"; cp=OUT/f"{fault}.csv"
    for i,(qid,q) in enumerate(QUESTIONS,1):
        print(f"\n[{i}/10] {qid}: {q}",flush=True)
        try:
            item=run_question(col,emb,fault,qid,q); data["questions"][qid]=item
            print(f"[RETRIEVED] {item['retrieved_count']} chunks")
            for h in item["retrieved"]:
                print(f"  rank={h['rank']} page={h['page']} source={h['source_file']} distance={h['distance']}")
                print(f"  preview={h['preview'][:160]}")
            print(f"[ANSWER] {item['answer']}")
            print(f"[METADATA] {'OK' if item['metadata_integrity'] else 'BROKEN'}")
        except Exception as e:
            err=f"{type(e).__name__}: {e}"
            data["questions"][qid]={"question":q,"answer":"","status":"ERROR","retrieved_count":0,
                "retrieved":[],"metadata_integrity":False,"error":err}
            print(f"[ERROR] {qid}: {err}")
        jp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
        with cp.open("w",newline="",encoding="utf-8-sig") as f:
            fields=["question_id","question","answer","status","retrieved_count","pages","sources","metadata_integrity","error"]
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
            for k,v in data["questions"].items():
                w.writerow({"question_id":k,"question":v.get("question",""),"answer":v.get("answer",""),
                    "status":v.get("status",""),"retrieved_count":v.get("retrieved_count",0),
                    "pages":";".join(str(h["page"]) for h in v.get("retrieved",[])),
                    "sources":";".join(str(h["source_file"]) for h in v.get("retrieved",[])),
                    "metadata_integrity":v.get("metadata_integrity",False),"error":v.get("error","")})
        print(f"[CHECKPOINT] {len(data['questions'])}/10 saved -> {jp.name}",flush=True)
    errors=sum(v.get("status")=="ERROR" for v in data["questions"].values())
    print(f"\n[SUMMARY] fault={fault} completed={10-errors}/10 errors={errors}")
    print(f"JSON={jp}\nCSV={cp}")
    return 1 if errors else 0

def main():
    p=argparse.ArgumentParser(description="W7D4 isolated RAG fault injection")
    p.add_argument("--fault",required=True,choices=list(DESCRIPTIONS))
    return run(p.parse_args().fault)

if __name__=="__main__": sys.exit(main())
