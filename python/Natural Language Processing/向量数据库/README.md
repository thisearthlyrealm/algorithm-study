# 上机任务二：基于向量数据库的企业级知识库构建实战

## 文件结构
- `data/`：实验数据，包含 Markdown、CSV、HTML 三种来源。
- `src/vector_kb_experiment.py`：完整实验源代码。
- `results/`：运行结果，包括检索结果 JSON、实体关系三元组 CSV、关键词词频图、知识图谱图。
- `report/上机任务二实验报告.docx`：实验报告。

## 运行方法
```bash
cd vector_db_assignment
python src/vector_kb_experiment.py
```

## 主要依赖
```bash
pip install numpy scikit-learn matplotlib networkx python-docx
```
可选依赖：`sentence-transformers`、`faiss-cpu`。若未安装，代码会自动使用本地 TF-IDF+SVD 和 NumPy 矩阵索引，保证可运行。
