import json, pandas as pd
from config import OUT_DIR, LABELED_CSV

# ---------- 读取结果 ----------
metrics = json.loads((OUT_DIR / "metrics.json").read_text(encoding="utf-8"))
comp = pd.read_csv(OUT_DIR / "feature_comparison.csv")
labeled = pd.read_csv(LABELED_CSV)

# ---------- 1. 终端报告 ----------
LINE = "=" * 68
print(LINE)
print("鄉村 / 城市 段落分類 — 結果總覽")
print(LINE)
print(f"標註總數 {len(labeled)} | 鄉村 {(labeled.label==0).sum()} | 城市 {(labeled.label==1).sum()}")
print(f"來源: {labeled.source.value_counts().to_dict()}\n")

print("【特徵比較】")
print(comp.to_string(index=False))
print()

best = max(metrics, key=lambda m: m["macro_f1"])
print(f"最佳特徵: {best['name']}  (Macro-F1 = {best['macro_f1']})\n")

for m in metrics:
    print(f"--- {m['name']} ---")
    print(f"準確率 {m['accuracy']}  Macro-F1 {m['macro_f1']}")
    cm = m["confusion_matrix"]
    print(f"混淆矩陣  [[{cm[0][0]:>3}, {cm[0][1]:>3}],")
    print(f"          [{cm[1][0]:>3}, {cm[1][1]:>3}]]")
    for cls, v in m["per_class"].items():
        print(f"  {cls}: P={v['precision']}  R={v['recall']}  F1={v['f1']}")
    print()

# ---------- 2. 生成 report.md 骨架 ----------
def cm_table(cm):
    return (f"| 真實 \\ 預測 | 鄉村 | 城市 |\n|---|---|---|\n"
            f"| 鄉村 | {cm[0][0]} | {cm[0][1]} |\n"
            f"| 城市 | {cm[1][0]} | {cm[1][1]} |")

def md_table(df):
    # 无依赖的 markdown 表生成（替代 to_markdown，避免引入 tabulate）
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(out)

md = ["# 鄉村 / 城市 段落分類：以《邊城》與《駱駝祥子》為例\n"]
md.append("## 1. 研究問題\n")
md.append("中文小說段落的敘事空間能否用詞袋模型區分為「鄉村」與「城市」兩類？"
          "本報告以《邊城》（鄉村）與《駱駝祥子》（城市）為語料，"
          "在段落層級訓練 LogisticRegression 分類器。\n")

md.append("## 2. 資料與標註\n")
md.append(f"- 語料：《邊城》《駱駝祥子》\n"
          f"- 文本單位：切分段（去 BOM → 空行分塊 → 句末切句 → 貪心合併），長度 50–400 字\n"
          f"- 標註總數：{len(labeled)}（鄉村 {(labeled.label==0).sum()} / 城市 {(labeled.label==1).sum()}）\n"
          f"- 標註方式：按書名打標（《邊城》=鄉村、《駱駝祥子》=城市）；"
          f"另設 `human_overrides.csv` 人工校正機制（改標/剔除後重跑 02 重新平衡）\n"
          f"- 類別平衡：對較大一類下採樣至 1:1\n")

md.append("## 3. 方法\n")
md.append("- 分詞：jieba，過濾單字\n"
          "- 特徵：詞頻 top-200 / 詞級 TF-IDF(1,2) / 字符級 TF-IDF(2,3)\n"
          "- 分類器：LogisticRegression（liblinear, class_weight=balanced）\n"
          "- 切分：80/20 分層抽樣，random_state=42\n")

md.append("## 4. 結果\n")
md.append("### 特徵比較\n")
md.append(md_table(comp))
md.append(f"\n最佳特徵為 **{best['name']}**，Macro-F1 = **{best['macro_f1']}**，"
          f"準確率 = **{best['accuracy']}**。\n")

md.append("### 混淆矩陣（最佳特徵）\n")
md.append(cm_table(best["confusion_matrix"]) + "\n")

md.append("### 各類別指標（最佳特徵）\n")
md.append("| 類別 | Precision | Recall | F1 |\n|---|---|---|---|")
for cls, v in best["per_class"].items():
    md.append(f"| {cls} | {v['precision']} | {v['recall']} | {v['f1']} |")
md.append("")

md.append("## 5. 核心局限\n")
md.append("兩部小說、兩位作者、兩個標籤完全共線。模型學到的可能是"
          "「沈從文文體 vs 老舍文體」，而非「鄉村敘事 vs 城市敘事」。"
          "報告須以此為核心局限討論，並建議未來引入同作者內部的城鄉對照段落來解耦。\n")

(OUT_DIR / "report.md").write_text("\n".join(md), encoding="utf-8")
print(f"[OK] 已寫入 {OUT_DIR / 'report.md'}")

# ---------- 3. 生成 results.html ----------
rows = ""
for m in metrics:
    cm = m["confusion_matrix"]
    rows += f"""
    <tr>
      <td>{m['name']}</td>
      <td class="num">{m['accuracy']}</td>
      <td class="num">{m['macro_f1']}</td>
      <td class="num">{m['per_class']['乡村']['f1']}</td>
      <td class="num">{m['per_class']['城市']['f1']}</td>
      <td class="cm">[{cm[0][0]}, {cm[0][1]}]<br>[{cm[1][0]}, {cm[1][1]}]</td>
    </tr>"""

html = f"""<!DOCTYPE html><html lang="zh-Hant"><head><meta charset="utf-8">
<title>鄉村/城市分類結果</title>
<style>
body{{background:#0f1115;color:#e8eaf0;font-family:system-ui,sans-serif;padding:40px;max-width:900px;margin:auto}}
h1{{font-size:24px}} h2{{color:#58a6ff;font-size:17px;margin-top:32px}}
table{{width:100%;border-collapse:collapse;font-size:13.5px;margin-top:12px}}
th,td{{padding:9px 12px;border-bottom:1px solid #262b36;text-align:left}}
th{{color:#9aa3b2;font-size:12.5px}} .num{{font-variant-numeric:tabular-nums}}
.cm{{font-family:monospace;color:#9aa3b2;font-size:12px}}
.best{{color:#3fb950;font-weight:700}}
</style></head><body>
<h1>鄉村 / 城市 段落分類結果</h1>
<p>語料：《邊城》《駱駝祥子》｜標註 {len(labeled)} 段｜分層 80/20｜random_state=42</p>
<h2>特徵比較</h2>
<table><tr><th>特徵</th><th>準確率</th><th>Macro-F1</th><th>鄉村 F1</th><th>城市 F1</th><th>混淆矩陣</th></tr>{rows}</table>
<p style="color:#9aa3b2;font-size:13px;margin-top:20px">
最佳特徵：<span class="best">{best['name']}</span>（Macro-F1 = {best['macro_f1']}）
</p></body></html>"""

(OUT_DIR / "results.html").write_text(html, encoding="utf-8")
print(f"[OK] 已寫入 {OUT_DIR / 'results.html'}")
