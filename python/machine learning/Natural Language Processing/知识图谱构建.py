import os
import re
from collections import Counter
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from wordcloud import WordCloud
from matplotlib import font_manager
OUTPUT_DIR=Path(__file__).resolve().parent.parent
OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
FONT_CANDIDATES=["C:/Windows/Fonts/simhei.ttf","C:/Windows/Fonts/msyh.ttc"]
FONT_PATH=next((p for p in FONT_CANDIDATES if os.path.exists(p)), None)
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
STOPWORDS = {
    "的","和","是","在","中","对","为","与","等","可","能","通过","进行","实现",
    "一种","重要","核心","领域","技术","任务","方法","系统","数据","模型","应用",
    "依赖","主要","通常","包括","具有","用于","能够","可以","需要","过程"
}
DOMAIN_TERMS = [
    "人工智能","机器学习","深度学习","监督学习","无监督学习","强化学习",
    "神经网络","卷积神经网络","循环神经网络","Transformer","大语言模型",
    "知识图谱","自然语言处理","计算机视觉","目标检测","图像分类",
    "语义分割","注意力机制","智能体","环境","学习策略","智能决策","实体识别","关系抽取","信息抽取","特征提取","模型训练",
    "标注数据","未标注数据","训练数据","测试数据","分类任务","聚类任务","回归预测",
    "推荐系统","智能问答","语义检索","文本分类","机器翻译","自动摘要",
    "实体","关系","三元组","节点","边","属性","语义关联","可视化",
    "准确率","召回率","F1值","参数调优","损失函数","优化算法","梯度下降",
    "过拟合","泛化能力","数据清洗","文本预处理","分词","词云图",
    "停用词调整","词典扩展","关系规则优化","可视化布局调整"
]
ENTITY_TYPES = {
    "人工智能": "领域", "机器学习": "子领域", "深度学习": "子领域", "监督学习": "学习范式",
    "无监督学习": "学习范式", "强化学习": "学习范式", "神经网络": "模型", "卷积神经网络": "模型",
    "循环神经网络": "模型", "Transformer": "模型", "大语言模型": "模型", "知识图谱": "技术",
    "自然语言处理": "应用方向", "计算机视觉": "应用方向", "目标检测": "任务", "图像分类": "任务",
    "语义分割": "任务", "注意力机制": "模型机制", "智能体": "强化学习要素", "环境": "强化学习要素",
    "学习策略": "强化学习要素", "智能决策": "应用场景", "实体识别": "信息抽取任务", "关系抽取": "信息抽取任务", "信息抽取": "技术",
    "特征提取": "方法", "模型训练": "过程", "标注数据": "数据", "未标注数据": "数据", "训练数据": "数据", "测试数据": "数据",
    "分类任务": "任务", "聚类任务": "任务", "回归预测": "任务", "推荐系统": "应用场景", "智能问答": "应用场景",
    "语义检索": "应用场景", "文本分类": "任务", "机器翻译": "任务", "自动摘要": "任务", "实体": "概念",
    "关系": "概念", "三元组": "数据结构", "节点": "图结构", "边": "图结构", "属性": "概念",
    "语义关联": "概念", "可视化": "结果形式", "准确率": "评价指标", "召回率": "评价指标", "F1值": "评价指标",
    "参数调优": "实验过程", "损失函数": "算法要素", "优化算法": "算法要素", "梯度下降": "优化算法",
    "过拟合": "问题", "泛化能力": "性能", "数据清洗": "预处理", "文本预处理": "预处理", "分词": "预处理",
    "词云图": "可视化结果", "停用词调整": "调优方法", "词典扩展": "调优方法",
    "关系规则优化": "调优方法", "可视化布局调整": "调优方法"
}
CORPUS = """
人工智能是研究机器模拟人类智能行为的重要领域。机器学习是人工智能的重要分支，强调从训练数据中学习规律并完成预测。
深度学习是机器学习的子领域，深度学习通过神经网络自动学习数据特征。卷积神经网络常用于计算机视觉中的图像分类、目标检测和语义分割。
循环神经网络曾被广泛用于自然语言处理中的文本分类、机器翻译和自动摘要。Transformer通过注意力机制提升自然语言处理效果，并推动大语言模型发展。
监督学习使用标注数据训练模型，常见任务包括分类任务和回归预测。无监督学习使用未标注数据发现隐藏结构，典型任务包括聚类任务。
强化学习通过智能体与环境交互学习策略，在推荐系统和智能决策中具有应用价值。
信息抽取是知识图谱构建的重要基础。信息抽取通常包括实体识别和关系抽取。实体识别从文本中发现人工智能、机器学习、神经网络等核心实体。
关系抽取挖掘实体之间的语义关联，并将结果组织为实体、关系、实体组成的三元组。
知识图谱由节点和边构成，节点表示实体，边表示关系，属性描述实体的补充信息。知识图谱可应用于智能问答、语义检索和推荐系统。
在实验过程中，需要先进行文本预处理和分词，再完成实体识别、关系抽取和知识图谱可视化。
为了评价抽取结果，可观察实体覆盖情况、关系正确性以及图谱结构是否清晰。参数调优主要包括停用词调整、词典扩展、关系规则优化和可视化布局调整。
模型训练通常依赖损失函数和优化算法，梯度下降是常见优化算法。过拟合会降低模型的泛化能力，因此需要使用测试数据进行验证。
""".strip()
def tokenize(text,domain_terms,stopwords):
    tokens=[]
    i=0
    while i<len(text):
        ch=text[i]
        if ch.isspace() or ch in "，。；：、（）()《》“”\"：；,.!?！？\n\t":
            i+=1
            continue
        m=re.match(r"[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)?",text[i:])
        if m:
            token=m.group(0)
            if token not in stopwords and len(token)>1:
                tokens.append(token)
            i+=len(token)
            continue
        matched=None
        for term in domain_terms:
            if text.startswith(term,i):
                matched=term
                break
        if matched:
            if matched not in stopwords:
                tokens.append(matched)
            i+=len(matched)
        else:
            i+=1
    return tokens
def extract_entities(text,entity_types):
    tokens=tokenize(text,DOMAIN_TERMS,set())
    counter=Counter(t for t in tokens if t in entity_types)
    records=[]
    for ent,count in counter.items():
        records.append({"实体":ent,"实体类型":entity_types.get(ent,"未知"),"出现次数":count})
    records.sort(key=lambda x:(-x["出现次数"],x["实体"]))
    return records
def split_sentences(text):
    return [s.strip() for s in re.split(r"[。！？；\n]+",text) if s.strip()]
def relation_by_keyword(sentence):
    rules=[
        ("是","属于/定义"),("分支","属于"),("子领域","属于"),("通过","依赖/通过"),("使用","使用"),("用于","应用于"),
        ("应用于","应用于"),("包括","包括"),("构成","构成"),("表示","表示"),("描述","描述"),("提升","提升"),
        ("推动","推动"),("挖掘","挖掘"),("组织为","组织为"),("评价","评价"),("降低","影响"),("依赖","依赖"),
    ]
    for key,rel in rules:
        if key in sentence:
            return rel
    return "相关"
def extract_relations(text,entity_types):
    triples=[]
    seen=set()
    entities=list(entity_types.keys())
    def add(head,relation,tail,evidence):
        if not head or not tail or head==tail:
            return
        key=(head,relation,tail)
        if key not in seen:
            triples.append({"头实体":head,"关系":relation,"尾实体":tail,"证据句":evidence})
            seen.add(key)
    def entity_occurrences(sentence):
        occ=[]
        for e in entities:
            start=0
            while True:
                idx=sentence.find(e,start)
                if idx==-1:
                    break
                covered_by_longer = any(
                    other!=e and len(other)>len(e) and sentence.find(other,max(0,idx-6),idx+len(e)+6)!=-1
                    and sentence.find(other,max(0,idx-6),idx+len(e)+6)<=idx
                    and sentence.find(other,max(0,idx-6),idx+len(e)+6)+len(other)>=idx+len(e)
                    for other in entities
                )
                if not covered_by_longer:
                    occ.append((idx,e))
                start=idx+1
        return sorted(occ,key=lambda x:x[0])
    def ents_in(sentence):
        seen_local=set()
        ordered=[]
        for _,e in entity_occurrences(sentence):
            if e not in seen_local:
                ordered.append(e)
                seen_local.add(e)
        return ordered
    def before_after(sentence,trigger):
        idx=sentence.find(trigger)
        if idx==-1:
            return [],[]
        before=[e for pos,e in entity_occurrences(sentence) if pos<idx]
        after=[e for pos,e in entity_occurrences(sentence) if pos>idx]
        return before,after
    for sentence in split_sentences(text):
        ents=ents_in(sentence)
        if len(ents)<2:
            continue
        if "是" in sentence and ("分支" in sentence or "子领域" in sentence):
            before,after=before_after(sentence,"是")
            if before and after:
                add(before[-1],"属于",after[0],sentence)
        if "是" in sentence and "基础" in sentence:
            before,after=before_after(sentence,"是")
            if before and after:
                add(before[-1],"支撑",after[0],sentence)
        if "通过" in sentence:
            before,after=before_after(sentence,"通过")
            if before and after:
                add(before[-1],"依赖/通过",after[0],sentence)
            if "提升" in sentence:
                b2,a2=before_after(sentence,"提升")
                if b2 and a2:
                    subject=before[-1] if before else b2[-1]
                    add(subject,"提升",a2[0],sentence)
            if "推动" in sentence:
                b2,a2=before_after(sentence,"推动")
                if b2 and a2:
                    subject=before[-1] if before else b2[-1]
                    add(subject,"推动",a2[0],sentence)
            if "应用价值" in sentence and "在" in sentence:
                subject=before[-1] if before else ents[0]
                _,app_after=before_after(sentence,"在")
                for tail in app_after:
                    if entity_types.get(tail)=="应用场景":
                        add(subject,"应用于",tail,sentence)
        if "使用" in sentence:
            before,after=before_after(sentence,"使用")
            if before and after:
                add(before[-1],"使用",after[0],sentence)
        for trigger in ["应用于","用于"]:
            if trigger=="用于" and "应用于" in sentence:
                continue
            if trigger in sentence:
                before,after=before_after(sentence,trigger)
                if before and after:
                    if trigger=="应用于":
                        for tail in after:
                            add(before[-1],"应用于",tail,sentence)
                    else:
                        add(before[-1],"应用于",after[0],sentence)
                        if len(after)>1:
                            for tail in after[1:]:
                                add(after[0],"包含",tail,sentence)
        if "包括" in sentence:
            before,after=before_after(sentence,"包括")
            if before and after:
                subject=before[0] if "任务包括" in sentence else before[-1]
                for tail in after:
                    add(subject,"包括",tail,sentence)
        if "由" in sentence and "构成" in sentence:
            before,after=before_after(sentence,"由")
            if before and after:
                for tail in after:
                    add(before[-1],"构成",tail,sentence)
        for trigger,rel in [("表示","表示"),("描述","描述")]:
            if trigger in sentence:
                before,after=before_after(sentence,trigger)
                if before and after:
                    add(before[-1],rel,after[0],sentence)
        if "挖掘" in sentence:
            before,after=before_after(sentence,"挖掘")
            if before and after:
                add(before[-1],"挖掘",after[0],sentence)
        if "组织为" in sentence:
            before,after=before_after(sentence,"组织为")
            if before and after:
                add(before[-1],"组织为",after[-1],sentence)
        if "依赖" in sentence:
            before,after=before_after(sentence,"依赖")
            if before and after:
                add(before[-1],"依赖",after[0],sentence)
        if "降低" in sentence:
            before,after=before_after(sentence,"降低")
            if before and after:
                add(before[-1],"影响",after[0],sentence)
    return triples
def draw_wordcloud(freq):
    wc=WordCloud(font_path=FONT_PATH,width=1200,height=800,background_color='white',max_words=80,collocations=False,prefer_horizontal=0.9,random_state=42).generate_from_frequencies(freq)
    out=OUTPUT_DIR/"wordcloud.png"
    wc.to_file(str(out))
    return out
def draw_knowledge_graph(triples,entity_types):
    G=nx.DiGraph()
    for tri in triples:
        h,r,t=tri["头实体"],tri["关系"],tri["尾实体"]
        G.add_node(h,node_type=entity_types.get(h,"未知"))
        G.add_node(t,node_type=entity_types.get(t,"未知"))
        G.add_edge(h,t,label=r)
    plt.figure(figsize=(15,10))
    pos=nx.spring_layout(G,k=1.1,seed=17,iterations=120)
    node_sizes=[900+90*G.degree(n) for n in G.nodes()]
    nx.draw_networkx_nodes(G,pos,node_size=node_sizes,alpha=0.92,linewidths=1.2,edgecolors="#555555")
    nx.draw_networkx_edges(G,pos,arrows=True,arrowstyle="-|>",arrowsize=18,width=1.2,alpha=0.65)
    nx.draw_networkx_labels(G,pos,font_size=10,font_family=plt.rcParams["font.sans-serif"][0])
    edge_labels=nx.get_edge_attributes(G,"label")
    nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels,font_size=8,font_family=plt.rcParams["font.sans-serif"][0],rotate=False)
    plt.title("人工智能与机器学习领域知识图谱",fontsize=18,pad=20)
    plt.axis("off")
    plt.tight_layout()
    out=OUTPUT_DIR/"knowledge_graph.png"
    plt.savefig(out,dpi=220,bbox_inches="tight")
    plt.close()
    return out
def main():
    (OUTPUT_DIR/"domain_corpus.txt").write_text(CORPUS,encoding="utf-8")
    tokens=tokenize(CORPUS,DOMAIN_TERMS,STOPWORDS)
    freq=Counter(tokens)
    freq_df=pd.DataFrame(freq.most_common(),columns=["词语","词频"])
    freq_df.to_csv(OUTPUT_DIR/"token_frequency.csv",index=False,encoding="utf-8-sig")
    entities=extract_entities(CORPUS,ENTITY_TYPES)
    end_df=pd.DataFrame(entities)
    end_df.to_csv(OUTPUT_DIR/"entities.csv",index=False,encoding="utf-8-sig")
    triples=extract_relations(CORPUS,ENTITY_TYPES)
    tri_df=pd.DataFrame(triples)
    tri_df.to_csv(OUTPUT_DIR/"relations_triples.csv",index=False,encoding="utf-8-sig")
    wordcloud_path=draw_wordcloud(freq)
    kg_path=draw_knowledge_graph(triples,ENTITY_TYPES)
    print("实验运行完成！")
    print(f"分词总数：{len(tokens)}，去重词数：{len(freq)}")
    print(f"识别实体数：{len(entities)}")
    print(f"抽取三元组数：{len(triples)}")
    print(f"词云图：{wordcloud_path}")
    print(f"知识图谱图：{kg_path}")
if __name__=="__main__":
    main()
