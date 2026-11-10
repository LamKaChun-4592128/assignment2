import os, json, re
import requests

BASE_API = "https://api.github.com/repos/mcjkurz/qhchina-data/contents/corpora/中文小说100强"
RAW_BASE = "https://raw.githubusercontent.com/mcjkurz/qhchina-data/main/corpora/中文小说100强"

RURAL_KEYWORDS = ["边城", ]
URBAN_KEYWORDS = ["骆驼祥子"]

def list_remote_files():
    r = requests.get(BASE_API, timeout=30)
    r.raise_for_status()
    return [item["name"] for item in r.json() if item["name"].endswith(".txt")]

def pick_files(all_files, keywords):
    picked = []
    for kw in keywords:
        for fname in all_files:
            if kw in fname and fname not in picked:
                picked.append(fname)
                break
    return picked

def download(fname, out_dir="data/raw"):
    os.makedirs(out_dir, exist_ok=True)
    url = f"{RAW_BASE}/{fname}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    path = os.path.join(out_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(r.text)
    print(f"✅ {fname} ({len(r.text)} 字)")
    return path

if __name__ == "__main__":
    all_files = list_remote_files()
    print("远端 .txt 文件数:", len(all_files))
    for fname in pick_files(all_files, RURAL_KEYWORDS + URBAN_KEYWORDS):
        download(fname)
