import json, jieba, numpy as np, pandas as pd, joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, classification_report)
from config import LABELED_CSV, OUT_DIR, SEED, TEST_SIZE, jieba_tokenize

tok = jieba_tokenize

FEATS = {
    "词频_top200": CountVectorizer(
        tokenizer=tok, max_features=200, token_pattern=None, lowercase=False),
    "词级TFIDF_1_2": TfidfVectorizer(
        tokenizer=tok, ngram_range=(1, 2), max_features=5000,
        token_pattern=None, lowercase=False, sublinear_tf=True),
    "字符TFIDF_2_3": TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 3), max_features=8000,
        sublinear_tf=True),
}

def run_one(name, vec, Xtr_t, Xte_t, ytr, yte):
    Xtr = vec.fit_transform(Xtr_t)
    Xte = vec.transform(Xte_t)
    clf = LogisticRegression(max_iter=3000, random_state=SEED,
                             class_weight="balanced", solver="liblinear")
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    p, r, f1, _ = precision_recall_fscore_support(yte, pred, labels=[0, 1])
    cm = confusion_matrix(yte, pred)
    return {
        "name": name,
        "accuracy": round(accuracy_score(yte, pred), 4),
        "macro_f1": round(float(f1.mean()), 4),
        "confusion_matrix": cm.tolist(),
        "per_class": {
            "乡村": {"precision": round(p[0],4), "recall": round(r[0],4), "f1": round(f1[0],4)},
            "城市": {"precision": round(p[1],4), "recall": round(r[1],4), "f1": round(f1[1],4)},
        },
        "report": classification_report(yte, pred, target_names=["乡村","城市"], digits=3),
        "_model": (vec, clf),
        "_cm": cm,
    }

def plot_cm(cm, title, path):
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["乡村","城市"]); ax.set_yticklabels(["乡村","城市"])
    ax.set_xlabel("预测"); ax.set_ylabel("真实")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i][j], ha="center", va="center",
                    color="white" if cm[i][j] > cm.max()/2 else "black")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=.8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

def main():
    df = pd.read_csv(LABELED_CSV)
    X_tr, X_te, y_tr, y_te = train_test_split(
        df.text.tolist(), df.label.tolist(),
        test_size=TEST_SIZE, random_state=SEED, stratify=df.label)
    print(f"训练 {len(X_tr)} / 测试 {len(X_te)}")

    results, best = [], None
    for name, vec in FEATS.items():
        res = run_one(name, vec, X_tr, X_te, y_tr, y_te)
        print(f"\n=== {name} ===  acc={res['accuracy']}  macroF1={res['macro_f1']}")
        print(res["report"])
        plot_cm(res["_cm"], name, OUT_DIR / f"cm_{name}.png")
        if best is None or res["macro_f1"] > best["macro_f1"]:
            best = res
        results.append({k: v for k, v in res.items() if not k.startswith("_")})

    joblib.dump({"vectorizer": best["_model"][0], "classifier": best["_model"][1]},
                OUT_DIR / "best_model.joblib")
    print(f"\n最佳特征: {best['name']}  macroF1={best['macro_f1']}")

    (OUT_DIR / "metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([{
        "特征": r["name"], "准确率": r["accuracy"], "Macro-F1": r["macro_f1"],
        "乡村F1": r["per_class"]["乡村"]["f1"], "城市F1": r["per_class"]["城市"]["f1"],
    } for r in results]).to_csv(
        OUT_DIR / "feature_comparison.csv", index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()
