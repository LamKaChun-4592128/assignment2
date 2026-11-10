# 计划：annotate.py 改为按书名打标

## 已确认决策
- 标签来源：按书名打标（2_边城.txt→0 乡村，3_骆驼祥子.txt→1 城市），不再用词典弱标注
- 语料规模：维持 2 本书（data/raw/ 仅这两本），不恢复已删除的 4 本

## 背景问题
原脚本按空行切块后过滤 50–400 字，但语料每章为一整行（边城中位 ~2000 字，骆驼祥子 ~5600 字），导致候选段落几乎为 0（1/48）。需重写切分逻辑。

## 实施步骤
1. 新建 `scripts/annotate.py`：
   - `BOOK_LABELS = {"边城": 0, "骆驼祥子": 1}`，文件名子串匹配打标
   - `split_paragraphs`：去 BOM（\ufeff）→ 空行分块 → 句末标点（。！？）切句 → 贪心合并为 50–400 字段落；单句 >400 硬切，尾段 <50 丢弃
   - 移除 jieba 与词典依赖（按书打标后不再需要）
   - 平衡抽样：显式对两类各 `sample(n=min, random_state=42)` 后 concat 并 shuffle（规避 pandas 3.1 groupby.apply 兼容问题）
   - 输出 `data/annotated.csv`（列：text, label, source），打印类别与来源分布
2. 运行 `python3 scripts/annotate.py`
3. 验证：行数、两类均衡（各 ~142）、抽查段落文本完整性

## 预期产出
- `data/annotated.csv` ≈ 284 行（乡村 142 + 城市 142）
- 环境：jieba 不再需要；pandas 3.1.1 已装
