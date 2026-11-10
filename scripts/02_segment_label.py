import os, re
import pandas as pd

BOOK_LABELS = {"边城": 0, "骆驼祥子": 1}  # 0=乡村, 1=城市
OVERRIDES_CSV = "data/human_overrides.csv"

def split_paragraphs(text, min_len=50, max_len=400):
    # 去 BOM；按空行分块，块内按句末标点切句，再贪心合并为 min_len~max_len 的段落
    text = text.replace("\ufeff", "")
    sentences = []
    for block in re.split(r"\n\s*\n|\r\n\s*\r\n", text):
        block = re.sub(r"\s+", "", block)
        if not block:
            continue
        sentences.extend(s for s in re.findall(r"[^。！？]+[。！？]?", block) if s)
    paras, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) <= max_len:
            cur += s
            continue
        if len(cur) >= min_len:
            paras.append(cur)
        if len(s) > max_len:  # 单句超长则硬切
            for i in range(0, len(s), max_len):
                piece = s[i:i + max_len]
                if len(piece) >= min_len:
                    paras.append(piece)
            cur = ""
        else:
            cur = s
    if len(cur) >= min_len:
        paras.append(cur)
    return paras

def label_book(fname):
    for key, label in BOOK_LABELS.items():
        if key in fname:
            return label
    return None

def load_overrides(path=OVERRIDES_CSV):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, dtype={"seg_id": str, "label": int})
    return dict(zip(df["seg_id"], df["label"]))

def apply_overrides(df, overrides):
    relabel = {k: v for k, v in overrides.items() if v in (0, 1)}
    drop = {k for k, v in overrides.items() if v == -1}
    before = df["label"].copy()
    df["label"] = df["seg_id"].map(relabel).fillna(df["label"]).astype(int)
    changed = int((df["label"] != before).sum())
    dropped = int(df["seg_id"].isin(drop).sum())
    df = df[~df["seg_id"].isin(drop)].reset_index(drop=True)
    print(f"🔧 人工校正：改标 {changed} 段，剔除 {dropped} 段")
    return df

def build_annotated(raw_dir="data/raw", out_csv="data/annotated.csv",
                    overrides_csv=OVERRIDES_CSV):
    rows = []
    for fname in sorted(os.listdir(raw_dir)):
        if not fname.endswith(".txt"):
            continue
        label = label_book(fname)
        if label is None:
            continue
        text = open(os.path.join(raw_dir, fname), encoding="utf-8").read()
        for i, para in enumerate(split_paragraphs(text)):
            rows.append({"seg_id": f"{fname}#{i:03d}", "text": para,
                         "label": label, "source": fname})
    df = pd.DataFrame(rows)

    overrides = load_overrides(overrides_csv)
    if overrides is None:
        # 首次运行生成人工校正模板（含自动标签）；此后重跑不会覆盖人工修改
        df.to_csv(overrides_csv, index=False)
        print(f"📝 已生成人工校正模板 {overrides_csv}")
        print("   用法：将 label 改为 0（乡村）/1（城市），-1 表示剔除该段，改后重跑本脚本")
    else:
        df = apply_overrides(df, overrides)

    # 类别平衡：两类各随机抽取较小类的数量，整体打乱
    n = min((df["label"] == 0).sum(), (df["label"] == 1).sum())
    df = pd.concat([
        df[df["label"] == 0].sample(n=n, random_state=42),
        df[df["label"] == 1].sample(n=n, random_state=42),
    ]).sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    print(f"📦 标注段落总数 {len(df)}，乡村 {(df.label==0).sum()}，城市 {(df.label==1).sum()}")
    print(f"   来源分布：\n{df.source.value_counts().to_string()}")
    return df

if __name__ == "__main__":
    build_annotated()
