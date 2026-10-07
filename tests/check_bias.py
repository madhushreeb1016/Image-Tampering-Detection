import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/features.csv")
meta = [c for c in df.columns if c in ("filename", "label")]
all_feats = [c for c in df.columns if c not in meta and df[c].dtype != object]
no_raw = [c for c in all_feats if "raw" not in c.lower()]
print("Dropped for 'no raw' run:", sorted(set(all_feats) - set(no_raw)))

df["grp"] = df.filename.str.extract(r"_(\d)\d{4}\.jpg")[0]
cv = StratifiedKFold(5, shuffle=True, random_state=42)

for name, cols in (("all features", all_feats), ("without raw features", no_raw)):
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    df["pred"] = cross_val_predict(model, df[cols].values, df.label.values, cv=cv)
    print(f"\n{name}: out-of-sample accuracy = {(df.pred == df.label).mean():.3f}")
    au = df[df.label == 0]
    print(au.groupby("grp").pred.agg(["count", "mean"]).rename(columns={"mean": "flagged_tampered"}).round(3))