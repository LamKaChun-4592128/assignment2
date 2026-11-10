# 计划：完成 annotate 步骤 + train.py 训练评估

## 前置决策（用户已确认）
- annotate.py 按书名打标（边城→0 乡村，骆驼祥子→1 城市），维持 2 本书
- 计划存档：.opencode/plans/annotate-book-label.md

## 步骤 A：补完 annotate（此前已被批准但被 plan 模式拦截）
1. 新建 `scripts/annotate.py`（内容见 annotate-book-label.md：去 BOM、句末标点切句、贪心合并 50–400 字、按书打标、显式平衡抽样、输出 data/annotated.csv）
2. 运行，预期 ~284 行（乡村 142 + 城市 142）

## 步骤 B：落地 train.py（用户提供的脚本 + 一处修复）
1. 新建 `scripts/train.py`，内容与用户提供版本一致，唯一修改：main 开头加 `os.makedirs("output", exist_ok=True)`（否则写 metrics.json 报 FileNotFoundError）
2. 依赖已就绪（sklearn 1.9.1 / jieba / pandas），无需安装

## 步骤 C：运行与验证
1. `python3 scripts/train.py`
2. 输出三组特征对比（freq_top200 / tfidf_word_1_2 / tfidf_char_2_3）
3. 产物：`output/metrics.json`、`output/feature_comparison.csv`
4. 汇报准确率、Macro-F1、混淆矩阵与特征比较表

## 注意事项
- 数据按书打标且来自同一本书，评估分数预计极高（≈作者分类器），解读时需说明
- jieba 首次运行会在临时目录写缓存，属正常现象
