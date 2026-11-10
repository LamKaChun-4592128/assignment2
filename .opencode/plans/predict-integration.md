# 计划补充：predict.py 集成与 final 模型产出

## 已确认决策（本轮）
- final_model.joblib 由 train.py main 末尾自动保存（惰性导入 predict.train_final_and_save，避免循环导入）
- final 模型用**全部标注数据**（train+test ≈284 行）重训；评估指标仍只来自 train split，口径不受污染
- predict.py 补一段乡村向 demo 对照（手写拟作，非训练集原文）

## scripts/predict.py 落地内容（基于用户版本，2 处修改）
1. `train_final_and_save` 内 dump 前加 `os.makedirs(os.path.dirname(path), exist_ok=True)`（需 `import os`）
2. `__main__` 增加乡村 demo（渡口/艄公/稻田/牛/溪拟作段），与《长恨歌》城市 demo 双双预测并打印对比
3. 其余保持原样（`from train import jieba_tokenize` 在 `python3 scripts/predict.py` 下可行；prob[1]=P(城市) 正确）

## scripts/train.py 追加（main 末尾）
```python
df_all = pd.read_csv("data/annotated.csv")
from predict import train_final_and_save
path = train_final_and_save(df_all["text"].tolist(), df_all["label"].tolist())
print(f"💾 最终模型已保存：{path}")
```

## 完整执行顺序（含前一轮计划）
1. 新建 `scripts/annotate.py`（按书名打标，见 annotate-book-label.md）→ 运行 → data/annotated.csv（~284 行）
2. 新建 `scripts/train.py`（用户版 + `os.makedirs("output", exist_ok=True)` + 上述 final 模型追加）→ 运行 → output/metrics.json、output/feature_comparison.csv、output/final_model.joblib
3. 新建 `scripts/predict.py`（用户版 + 2 处修改）→ 运行 → 打印两个 demo 的预测概率
4. 汇总汇报：三组特征对比表、demo 预测结果（P城市/P乡村）

## 状态提醒
- 当前环境：sklearn 1.9.1 / jieba 0.42.1 / pandas 3.1.1 / joblib 1.6.0 均就绪，无需安装
- 仍处 plan 模式，等待用户退出后一次性执行
