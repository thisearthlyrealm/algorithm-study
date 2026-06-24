import csv
import html.parser
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict,List,Tuple
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances
from sklearn.decomposition import TruncatedSVD
import warnings
warnings.filterwarnings('ignore')
BASE_DIR=Path(__file__).resolve().parents[1]
DATA_DIR=BASE_DIR/'data'
RESULT_DIR=BASE_DIR/'results'
RESULT_DIR.mkdir(exist_ok=True)
@dataclass
class Document:
    source:str
    title:str
    text:str
@dataclass
class Chunk:
    text:str
    metadata:Dict[str,object]
class SimpleHTMLTextExtractor(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts=[]
        self.current_title='网页内容'
    def handle_starttag(self,tag,attrs):
        if tag in {'h1','h2','h3','p','br'}:
            self.parts.append('\n')
    def handle_data(self,data):
        text=data.strip()
        if text:
            self.parts.append(text)
    def get_text(self):
        return '\n'.join([p.strip() for p in self.parts if p.strip()])
def clean_text(text:str)->str:
    text=unicodedata.normalize('NFKC',text)
    text=text.encode('utf-8',errors='ignore').decode('utf-8',errors='ignore')
    text=re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]','',text)
    text=re.sub(r'[★◆●▲■□◇※]+','',text)
    lines=[]
    seen=set()
    for line in text.splitlines():
        line=re.sub(r'\s+',' ',line).strip()
        if not line or len(line)<4:
            continue
        if line in seen:
            continue
        seen.add(line)
        lines.append(line)
    return '\n'.join(lines)
def load_markdown(path:Path)->List[Document]:
    raw=path.read_text(encoding='utf-8')
    text=clean_text(raw)
    docs=[]
    current_title='未命名章节'
    current=[]
    for line in text.splitlines():
        if line.startswith('#'):
            if current:
                docs.append(Document(path.name,current_title,'\n'.join(current)))
                current=[]
            current_title=line.strip('#').strip()
        else:
            current.append(line)
    if current:
        docs.append(Document(path.name,current_title,'\n'.join(current)))
    return docs
def load_csv(path:Path)->List[Document]:
    docs=[]
    with path.open('r',encoding='utf-8') as f:
        for row in csv.DictReader(f):
            title=f"{row['source']}-{row['category']}"
            docs.append(Document(path.name,title,clean_text(row['text'])))
    return docs
def load_html(path:Path)->List[Document]:
    parser=SimpleHTMLTextExtractor()
    parser.feed(path.read_text(encoding='utf-8'))
    text=clean_text(parser.get_text())
    docs=[]
    current='网页内容'
    buf=[]
    for line in text.splitlines():
        if line.endswith('说明') or line.startswith('向量数据库'):
            if buf:
                docs.append(Document(path.name,current,'\n'.join(buf)))
            current=line
            buf=[]
        else:
            buf.append(line)
    if buf:
        docs.append(Document(path.name,current,'\n'.join(buf)))
    return docs
def load_documents()->List[Document]:
    docs=[]
    docs.extend(load_markdown(DATA_DIR/'company_profile.md'))
    docs.extend(load_csv(DATA_DIR/'supply_chain.csv'))
    docs.extend(load_html(DATA_DIR/'tech_notes.html'))
    return docs
def smart_chunk_documents(docs:List[Document],max_chars:int=120,overlap:int=25)->List[Chunk]:
    chunks=[]
    for doc in docs:
        sentences=re.split(r'(?<=[。！？.!?])',doc.text)
        buffer=''
        idx=0
        for sent in sentences:
            sent=sent.strip()
            if not sent:
                continue
            if len(buffer)+len(sent)<=max_chars:
                buffer+=sent
            else:
                if buffer:
                    chunks.append(Chunk(buffer,{'source':doc.source,'section':doc.title,'chunk_id':idx,'strategy':'paragraph_sentence'}))
                    idx+=1
                if len(sent)>max_chars:
                    start=0
                    while start<len(sent):
                        part=sent[start:start+max_chars]
                        chunks.append(Chunk(part,{'source':doc.source,'section':doc.title,'chunk_id':idx,'strategy':'sliding_window'}))
                        idx+=1
                        start+=max_chars-overlap
                    buffer=''
                else:
                    buffer=sent
        if buffer:
            chunks.append(Chunk(buffer,{'source':doc.source,'section':doc.title,'chunk_id':idx,'strategy':'paragraph_sentence'}))
    return chunks
def fixed_chunk(doc:Document,chunk_size:int=120)->List[Chunk]:
    """普通分块：按固定长度切割，无重叠"""
    chunks=[]
    for i in range(0,len(doc.text),chunk_size):
        part=doc.text[i:i+chunk_size]
        if part.strip():
            chunks.append(Chunk(part,{'source':doc.source,'section':doc.title,'chunk_id':len(chunks),'strategy':'fixed'}))
    return chunks
def overlap_chunk(doc:Document,chunk_size:int=120,overlap:int=25)->List[Chunk]:
    """重叠分块：滑动窗口切割，有重叠"""
    chunks=[]
    start=0
    while start<len(doc.text):
        part=doc.text[start:start+chunk_size]
        if part.strip():
            chunks.append(Chunk(part,{'source':doc.source,'section':doc.title,'chunk_id':len(chunks),'strategy':'overlap'}))
        start+=chunk_size-overlap
    return chunks
class EmbeddingModel:
    def __init__(self):
        self.model_type='tfidf_svd_local_fallback'
        self.model=None
        self.vectorizer=None
        self.svd=None
        try:
            from sentence_transformers import SentenceTransformer
            self.model=SentenceTransformer('BAAI/bge-small-zh-v1.5')
            self.model_type='BAAI/bge-small-zh-v1.5'
        except Exception:
            self.model=None
    def fit_transform(self,texts:List[str])->np.ndarray:
        if self.model is not None:
            vectors=self.model.encode(texts,normalize_embeddings=True)
            return np.asarray(vectors,dtype='float32')
        self.vectorizer=TfidfVectorizer(analyzer='char',ngram_range=(2,4),min_df=1)
        tfidf=self.vectorizer.fit_transform(texts)
        n_components=min(64,max(2,min(tfidf.shape)-1))
        self.svd=TruncatedSVD(n_components=n_components,random_state=42)
        vectors=self.svd.fit_transform(tfidf)
        return vectors.astype('float32')
    def transform(self, texts: List[str]) -> np.ndarray:
        if self.model is not None:
            return np.asarray(self.model.encode(texts, normalize_embeddings=True), dtype='float32')
        tfidf = self.vectorizer.transform(texts)
        return self.svd.transform(tfidf).astype('float32')
class VectorIndex:
    def __init__(self,vectors:np.ndarray,chunks:List[Chunk],keyword_vectorizer=None,keyword_doc_matrix=None):
        self.vectors=vectors.astype('float32')
        self.chunks=chunks
        self.backend='numpy_matrix_index'
        self.keyword_vectorizer=keyword_vectorizer
        self.keyword_doc_matrix=keyword_doc_matrix
        self.faiss_index_l2=None
        try:
            import faiss
            self.backend='FAISS_IndexFlatL2'
            self.faiss=faiss
            self.faiss_index_l2=faiss.IndexFlatL2(vectors.shape[1])
            self.faiss_index_l2.add(self.vectors)
        except Exception:
            self.faiss=None
    def search_cosine(self,query_vec:np.ndarray,top_k:int=3)->List[Dict]:
        sims=cosine_similarity(query_vec.reshape(1,-1),self.vectors)[0]
        order=np.argsort(-sims)[:top_k]
        return [self._pack(i,float(sims[i]),'cosine') for i in order]
    def search_l2(self,query_vec:np.ndarray,top_k:int=3)->List[Dict]:
        if self.faiss_index_l2 is not None:
            dists,ids=self.faiss_index_l2.search(query_vec.reshape(1,-1).astype('float32'),top_k)
            return [self._pack(int(i),float(d),'euclidean') for i,d in zip(ids[0],dists[0])]
        dists=euclidean_distances(query_vec.reshape(1,-1),self.vectors)[0]
        order=np.argsort(dists)[:top_k]
        return [self._pack(i,float(dists[i]),'euclidean') for i in order]
    def search_keyword(self,query_text:str,top_k:int=3)->List[Dict]:
        """关键词检索：TF-IDF 词袋匹配，不做语义扩展"""
        if self.keyword_vectorizer is None or self.keyword_doc_matrix is None:
            return [{'metric':'keyword','score':0.0,'text':'关键词检索未启用','metadata':{}}]
        qv=self.keyword_vectorizer.transform([query_text])
        sims=cosine_similarity(qv,self.keyword_doc_matrix)[0]
        order=np.argsort(-sims)[:top_k]
        return [self._pack(i,float(sims[i]),'keyword') for i in order if sims[i]>0]
    def search_ip(self,query_vec:np.ndarray,top_k:int=3)->List[Dict]:
        """IP 内积检索：值越大越相似（向量需归一化）"""
        sims=np.dot(self.vectors,query_vec)
        order=np.argsort(-sims)[:top_k]
        return [self._pack(i,float(sims[i]),'inner_product') for i in order]
    def _pack(self,i:int,score:float,metric:str)->Dict:
        return {'metric':metric,'score':round(score,4),'text':self.chunks[i].text,'metadata':self.chunks[i].metadata}
def extract_entities_relations(chunks:List[Chunk])->Tuple[List[Dict],List[Dict]]:
    entity_terms=['订单系统','仓储部门','质量追溯模块','供应链金融','物流模块','客户咨询','Embedding 模型','FAISS','Chroma','Milvus','Qdrant','企业级知识库','向量数据库','元数据','汽车零部件','华中智造科技有限公司']
    relations=[]
    entities=[]
    all_text='\n'.join(c.text for c in chunks)
    for term in entity_terms:
        if term in all_text:
            entities.append({'entity':term,'type':'业务/技术实体'})
    patterns=[('订单系统','记录','客户名称/车型/零部件编码'),('仓储部门','关注','库存数量/库位编码'),('质量追溯模块','关联','供应商/生产批次/质检结果'),('供应链金融','评估','融资风险'),('Embedding 模型','转换','高维向量'),('FAISS','支持','本地向量检索'),('企业级知识库','保留','来源与切片元数据')]
    for s,p,o in patterns:
        if s in all_text:
            relations.append({'subject':s,'predicate':p,'object':o})
    return entities,relations
def save_wordcloud_like(chunks:List[Chunk]):
    text=''.join(c.text for c in chunks)
    words=re.findall(r'[\u4e00-\u9fa5]{2,}|[A-Za-z]{2,}',text)
    stop={'需要','可以','通过','文本','企业','模块','结果','用于','包括','关注','来源'}
    cnt=Counter(w for w in words if w not in stop)
    common=cnt.most_common(20)
    plt.figure(figsize=(10,6))
    plt.bar([w for w,_ in common],[v for _,v in common])
    plt.xticks(rotation=45,ha='right')
    plt.tight_layout()
    plt.savefig(RESULT_DIR/'wordcloud_keywords.png',dpi=180)
    plt.close()
def save_kg(relations:List[Dict]):
    g=nx.DiGraph()
    for r in relations:
        g.add_edge(r['subject'],r['object'],label=r['predicate'])
    plt.figure(figsize=(10,7))
    pos=nx.spring_layout(g,seed=42,k=0.8)
    nx.draw(g,pos,with_labels=True,node_size=2000,font_size=9,arrows=True)
    edge_labels=nx.get_edge_attributes(g,'label')
    nx.draw_networkx_edge_labels(g,pos,edge_labels=edge_labels,font_size=8)
    plt.title('企业知识图谱可视化效果图')
    plt.tight_layout()
    plt.savefig(RESULT_DIR/'knowledge_graph.png',dpi=180)
    plt.close()
def generate_answer(question:str,results:List[Dict])->str:
    """基于 Top-K 检索结果生成自然语言回答"""
    if not results:
        return '抱歉，知识库中未找到相关信息。'
    best=results[0]
    parts=[f'根据知识库中【{best["metadata"]["section"]}】的内容：']
    parts.append(best['text'])
    extras=[r for r in results[1:] if r['score']>0.3]
    if extras:
        parts.append('\n补充参考：')
        for r in extras:
            parts.append(f'· {r["metadata"]["section"]}：{r["text"][:60]}...')
    return '\n'.join(parts)

def main():
    CHUNK_SIZE=120
    OVERLAP=25
    # ========== 任务一①：文本清洗前后对比 ==========
    print("="*60)
    print("【文本清洗前后对比】")
    print("="*60)
    # 从 3 种数据源各取一个样本
    for name in ['company_profile.md','supply_chain.csv','tech_notes.html']:
        raw=Path(DATA_DIR/name).read_text(encoding='utf-8')
        cleaned=clean_text(raw)
        print(f"\n数据源：{name}")
        print("── 清洗前（前 200 字）──")
        print(raw[:200])
        print("── 清洗后（前 200 字）──")
        print(cleaned[:200])
    # ========== 任务一②：普通分块 vs 重叠分块对比 ==========
    print("\n"+"="*60)
    print("【普通分块 vs 重叠分块 对比】")
    print("="*60)
    docs=load_documents()
    for i,doc in enumerate(docs[:3]):
        f_chunks=fixed_chunk(doc,CHUNK_SIZE)
        o_chunks=overlap_chunk(doc,CHUNK_SIZE,OVERLAP)
        print(f"\n[{i+1}] {doc.source} - {doc.title}")
        print(f"   普通分块（固定{CHUNK_SIZE}字）       → {len(f_chunks)} 个片段")
        print(f"   重叠分块（窗口{CHUNK_SIZE}字，重叠{OVERLAP}字） → {len(o_chunks)} 个片段")
        if f_chunks:
            print(f"   普通示例：{f_chunks[0].text[:60]}...")
        if o_chunks:
            print(f"   重叠示例：{o_chunks[0].text[:60]}...")
    # ========== 任务一③：分块总数统计 ==========
    print("\n"+"="*60)
    print("【分块总数统计】")
    print("="*60)
    all_fixed=[]
    all_overlap=[]
    for doc in docs:
        all_fixed.extend(fixed_chunk(doc,CHUNK_SIZE))
        all_overlap.extend(overlap_chunk(doc,CHUNK_SIZE,OVERLAP))
    print(f"\n文档总数：{len(docs)}")
    print(f"普通分块总数：{len(all_fixed)}")
    print(f"重叠分块总数：{len(all_overlap)}")
    src_fixed=Counter(c.metadata['source'] for c in all_fixed)
    src_overlap=Counter(c.metadata['source'] for c in all_overlap)
    print("\n按来源统计（普通分块）：")
    for src,cnt in src_fixed.most_common():
        print(f"  {src:30s} → {cnt} 个分块")
    print("\n按来源统计（重叠分块）：")
    for src,cnt in src_overlap.most_common():
        print(f"  {src:30s} → {cnt} 个分块")
    # ========== 后续流程（使用重叠分块继续）==========
    chunks=all_overlap
    model=EmbeddingModel()
    vectors=model.fit_transform([c.text for c in chunks])
    # 建关键词检索引擎（TF-IDF 词袋匹配）
    kw_vectorizer=TfidfVectorizer(analyzer='char',ngram_range=(2,4),min_df=1)
    kw_doc_matrix=kw_vectorizer.fit_transform([c.text for c in chunks])
    index=VectorIndex(vectors,chunks,keyword_vectorizer=kw_vectorizer,keyword_doc_matrix=kw_doc_matrix)
    # ========== 任务二：三类检索对比 ==========
    print("\n"+"="*60)
    print("【三类检索对比 - 同一问题】")
    print("="*60)
    task2_q='供应链金融如何评估融资风险？'
    qv=model.transform([task2_q])[0]
    print(f"\n问题：{task2_q}\n")
    print("--- ① 关键词检索（TF-IDF 词袋匹配）Top-3 ---")
    for r in index.search_keyword(task2_q,3):
        print(f"  [{r['score']:.4f}] {r['text'][:50]}...")
        print(f"           来源：{r['metadata']['section']}")
    print("\n--- ② L2 欧氏距离检索 Top-3 ---")
    for r in index.search_l2(qv,3):
        print(f"  [{r['score']:.4f}] {r['text'][:50]}...")
        print(f"           来源：{r['metadata']['section']}")
    print("\n--- ③ IP 内积检索 Top-3 ---")
    for r in index.search_ip(qv,3):
        print(f"  [{r['score']:.4f}] {r['text'][:50]}...")
        print(f"           来源：{r['metadata']['section']}")
    # ========== 任务三：知识库问答 ==========
    print("\n"+"="*60)
    print("【知识库问答 - 5组自定义问题】")
    print("="*60)
    print("\n请输入5个问题（逐行输入，回车提交）：\n")
    task3_questions=[]
    defaults=['如何查询汽车零部件库存和订单状态？','仓储管理关注哪些核心数据？','向量数据库有哪些常见技术选型？','企业建设知识库需要哪些步骤？','质量追溯模块如何保障产品质量？']
    for i in range(5):
        try:
            q=input(f'问题 {i+1}/5：').strip()
        except (EOFError,KeyboardInterrupt):
            q=''
        if not q:
            q=defaults[i]
            print(f'  （使用默认问题：{q}）')
        task3_questions.append(q)
    for i,q in enumerate(task3_questions,1):
        qv=model.transform([q])[0]
        results=index.search_cosine(qv,3)
        answer=generate_answer(q,results)
        print(f"\n{'='*55}")
        print(f"【问题 {i}】{q}")
        print(f"{'='*55}")
        print(f"\n【回答】{answer}")
        print(f"\n参考来源（Top-3）：")
        for j,r in enumerate(results,1):
            print(f"  [{j}] [{r['score']:.4f}] {r['metadata']['section']}")
            print(f"      {r['text'][:70]}...")
    # ========== 原后续流程 ==========
    questions=['如何查询汽车零部件库存和订单状态？','Embedding模型和向量数据库有什么作用？','供应链金融如何评估融资风险？']
    search_results=[]
    for q in questions:
        qv=model.transform([q])[0]
        search_results.append({'question':q,'top_k':index.search_cosine(qv,3)})
    compare_q='文本切片会怎样影响向量检索效果？'
    qv=model.transform([compare_q])[0]
    metric_compare={'question':compare_q,'cosine':index.search_cosine(qv,3),'euclidean':index.search_l2(qv,3)}
    entities,relations=extract_entities_relations(chunks)
    save_wordcloud_like(chunks)
    save_kg(relations)
    output={
        'document_count': len(docs),
        'chunk_count': len(chunks),
        'embedding_model': model.model_type,
        'vector_dimension': int(vectors.shape[1]),
        'vector_db_backend': index.backend,
        'search_results': search_results,
        'metric_compare': metric_compare,
        'entities': entities,
        'relations': relations
    }
    (RESULT_DIR/'experiment_results.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    with (RESULT_DIR/'entity_relation_triples.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['subject','predicate','object'])
        writer.writeheader(); writer.writerows(relations)
    print(json.dumps(output,ensure_ascii=False,indent=2))
if __name__ == '__main__':
    main()