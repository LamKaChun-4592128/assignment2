import joblib
from config import OUT_DIR

def load():
    obj = joblib.load(OUT_DIR / "best_model.joblib")
    return obj["vectorizer"], obj["classifier"]

def predict(text, vec, clf):
    prob = clf.predict_proba(vec.transform([text]))[0]
    return ("城市" if prob[1] >= 0.5 else "乡村"), float(prob[1]), float(prob[0])

if __name__ == "__main__":
    vec, clf = load()
    demos = [
        ("翠翠坐在溪边，望着溪面为暮色所笼罩的一切，且望到那只渡船上一群过渡人，"
         "其中有个吸旱烟的打着火镰吸烟，把烟杆在船边剥剥地敲着烟灰。", "《边城》风格"),
        ("祥子看着那些洋车，从西直门一直排到平则门，街上满是车夫、巡警、"
         "卖吃食的与闲逛的人，锣鼓声与车铃声混成一片。", "《骆驼祥子》风格"),
    ]
    for t, note in demos:
        label, pu, pr = predict(t, vec, clf)
        print(f"{note}\n  预测={label}  P(城市)={pu:.3f}  P(乡村)={pr:.3f}\n")
