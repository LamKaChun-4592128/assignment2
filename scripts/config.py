from pathlib import Path

import jieba
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager

LABELED_CSV = Path("data/annotated.csv")
OUT_DIR = Path("output")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42
TEST_SIZE = 0.2

def jieba_tokenize(text):
    # 去掉单字与标点；须放在可导入模块中，joblib 序列化的 tokenizer 按模块引用还原
    return [w for w in jieba.lcut(text) if len(w) > 1]

def _setup_cjk_font():
    # 注册仓库内 Noto Sans CJK，混淆矩阵图中文标签可正常渲染
    font_path = Path("assets/fonts/NotoSansCJKsc-Regular.otf")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
        matplotlib.rcParams["axes.unicode_minus"] = False
        return "Noto Sans CJK SC"
    return None

CJK_FONT = _setup_cjk_font()
