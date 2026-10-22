import os, sys, json, time
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

print("="*70)
print("  Parkinsons Telemonitoring: MLR Analysis")
print("="*70)

csv_path = "telemonitoring_parkinsons_updrs.data.csv"
df_raw = pd.read_csv(csv_path)
TARGET = "motor_UPDRS"
DROP_COLS = ["subject#", "total_UPDRS"]
df = df_raw.drop(columns=DROP_COLS).dropna()

X_raw = df.drop(columns=[TARGET])
y = df[TARGET]
feature_names = list(X_raw.columns)
p_total = len(feature_names)
print(f"Features ({p_total}): {feature_names}")

X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

cv = KFold(n_splits=5, shuffle=True, random_state=42)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_names, index=X_test.index)

method_feature_subsets = {}

# Pearson
print("[1/5] Pearson...")
corrs = X_train.apply(lambda col: np.abs(np.corrcoef(col, y_train)[0,1]))
pearson_ranked = list(corrs.sort_values(ascending=False).index)
method_feature_subsets["Pearson Correlation"] = [pearson_ranked[:k] for k in range(1, p_total+1)]
print(f"  Top 5: {pearson_ranked[:5]}")

# LassoCV
print("[2/5] LassoCV...")
lasso_m = LassoCV(cv=cv, random_state=42, max_iter=10000).fit(X_train_scaled, y_train)
lasso_ranked = [feature_names[i] for i in np.argsort(-np.abs(lasso_m.coef_))]
method_feature_subsets["LassoCV"] = [lasso_ranked[:k] for k in range(1, p_total+1)]
print(f"  Top 5: {lasso_ranked[:5]}")

# RFE
print("[3/5] RFE...")
rfe = RFE(estimator=LinearRegression(fit_intercept=True), n_features_to_select=1, step=1)
rfe.fit(X_train_scaled, y_train)
rfe_ranked = [x for _,x in sorted(zip(rfe.ranking_, feature_names))]
method_feature_subsets["RFECV"] = [rfe_ranked[:k] for k in range(1, p_total+1)]
print(f"  Top 5: {rfe_ranked[:5]}")

# Forward SFS
print("[4/5] Forward SFS...")
t0=time.time()
sfs_selected=[]
sfs_remaining=list(feature_names)
sfs_subsets=[]
name_to_idx={n:i for i,n in enumerate(feature_names)}
sfs_folds=[]
for tr_idx,val_idx in cv.split(X_train):
    sfs_folds.append((X_train.iloc[tr_idx].values, y_train.iloc[tr_idx].values,
                      X_train.iloc[val_idx].values, y_train.iloc[val_idx].values))
for k in range(1, p_total+1):
    best_c=None; best_rmse=float("inf")
    for feat in sfs_remaining:
        cidx=[name_to_idx[f] for f in sfs_selected]+[name_to_idx[feat]]
        mses=[]
        for Xtr,ytr,Xval,yval in sfs_folds:
            Atr=np.column_stack([np.ones(len(Xtr)), Xtr[:,cidx]])
            Aval=np.column_stack([np.ones(len(Xval)), Xval[:,cidx]])
            b=np.linalg.lstsq(Atr,ytr,rcond=None)[0]
            mses.append(np.mean((yval-Aval@b)**2))
        cr=np.sqrt(np.mean(mses))
        if cr<best_rmse: best_rmse=cr; best_c=feat
    sfs_selected.append(best_c); sfs_remaining.remove(best_c)
    sfs_subsets.append(list(sfs_selected))
    if k<=3 or k==p_total: print(f"  k={k}: '{best_c}' CV RMSE={best_rmse:.4f}")
method_feature_subsets["Forward SFS"]=sfs_subsets
print(f"  SFS done in {time.time()-t0:.1f}s")

# Baseline OLS
print("[5/5] Baseline OLS...")
Xtr_f=np.column_stack([np.ones(len(X_train)), X_train.values])
b_f=np.linalg.lstsq(Xtr_f, y_train.values, rcond=None)[0]
res_f=y_train.values-Xtr_f@b_f
s2_f=np.sum(res_f**2)/max(len(X_train)-p_total-1,1)
XtX_f=Xtr_f.T@Xtr_f+1e-10*np.eye(Xtr_f.shape[1])
se_f=np.sqrt(np.maximum(np.diag(s2_f*np.linalg.inv(XtX_f)),0))
ts_f=np.abs(b_f)/np.maximum(se_f,1e-14)
ft=[(feature_names[i],ts_f[i+1]) for i in range(p_total)]
ols_ranked=[f for f,_ in sorted(ft,key=lambda x:-x[1])]
method_feature_subsets["Baseline OLS"]=[ols_ranked[:k] for k in range(1, p_total+1)]
print(f"  Top 5: {ols_ranked[:5]}")

# Evaluate all combos
print("Evaluating all method x k...")
np.random.seed(42)

def eval_ols(X_tr, y_tr, X_te, y_te, cols, n_train):
    Xtr=np.column_stack([np.ones(len(X_tr)), X_tr[cols].values])
    Xte=np.column_stack([np.ones(len(X_te)), X_te[cols].values])
    XtX=Xtr.T@Xtr+1e-10*np.eye(Xtr.shape[1])
    try: iXtX=np.linalg.inv(XtX)
    except: iXtX=np.linalg.pinv(XtX)
    b=iXtX@(Xtr.T@y_tr.values)
    yp=Xte@b; res=y_te.values-yp
    k=len(cols); n=len(y_te)
    rmse=float(np.sqrt(np.mean(res**2)))
    r2_v=float(r2_score(y_te,yp))
    mae=float(np.mean(np.abs(res)))
    adj_r2=float(1-(1-r2_v)*(n-1)/max(n-k-1,1))
    yptr=Xtr@b; res_tr=y_tr.values-yptr
    df_e=max(len(X_tr)-k-1,1)
    s2=float(np.sum(res_tr**2)/df_e)
    sst=float(np.sum((y_te.values-np.mean(y_te.values))**2))
    sse=float(np.sum(res**2))
    f_s=float((sst-sse)/max(k,1)/max(s2,1e-14))
    se_b=np.sqrt(np.maximum(np.diag(s2*iXtX),0))
    ts=b/np.maximum(se_b,1e-14)
    df_t=max(n_train-k-1,1)
    pv=2*(1-stats.t.cdf(np.abs(ts),df=df_t))
    tc=float(stats.t.ppf(0.975,df=df_t))
    ci_lo=b-tc*se_b; ci_hi=b+tc*se_b
    vifs_vals=[1.0]
    if k>1:
        Xc=X_tr[cols].values; vifs_vals=[]
        for j in range(k):
            yj=Xc[:,j]; Xj=np.column_stack([np.ones(len(Xc)), Xc[:,[i for i in range(k) if i!=j]]])
            bj=np.linalg.lstsq(Xj,yj,rcond=None)[0]; pj=Xj@bj
            sres=np.sum((yj-pj)**2); stot=np.sum((yj-np.mean(yj))**2)
            r2j=1-sres/max(stot,1e-14); vifs_vals.append(float(1/max(1-r2j,1e-6)))
    coefs=[]
    for i,fn in enumerate(["Intercept"]+list(cols)):
        vv=None if i==0 else (vifs_vals[i-1] if i-1<len(vifs_vals) else None)
        coefs.append({"feature":fn,"coef":float(b[i]),"stderr":float(se_b[i]),"tstat":float(ts[i]),
                      "pvalue":float(pv[i]),"ci_lower":float(ci_lo[i]),"ci_upper":float(ci_hi[i]),"vif":vv})
    cv_rs=[]
    for ti,vi in cv.split(X_tr):
        At=np.column_stack([np.ones(len(ti)), X_tr.iloc[ti][cols].values])
        Av=np.column_stack([np.ones(len(vi)), X_tr.iloc[vi][cols].values])
        bv=np.linalg.lstsq(At,y_tr.iloc[ti].values,rcond=None)[0]
        cv_rs.append(np.sqrt(np.mean((y_tr.iloc[vi].values-Av@bv)**2)))
    cv_rmse=float(np.mean(cv_rs))
    cv_r2_list=[]
    for ti,vi in cv.split(X_tr):
        At=np.column_stack([np.ones(len(ti)), X_tr.iloc[ti][cols].values])
        Av=np.column_stack([np.ones(len(vi)), X_tr.iloc[vi][cols].values])
        bv=np.linalg.lstsq(At,y_tr.iloc[ti].values,rcond=None)[0]
        cv_r2_list.append(r2_score(y_tr.iloc[vi].values,Av@bv))
    cv_r2=float(np.mean(cv_r2_list))
    levs=np.sum((Xte@iXtX)*Xte,axis=1)
    mg=tc*np.sqrt(s2*(1+levs)); pil=yp-mg; piu=yp+mg
    ins=(y_te.values>=pil)&(y_te.values<=piu)
    pi_cov=float(np.mean(ins)*100); mpiw=float(np.mean(piu-pil))
    ns=min(300,len(X_te)); si=np.random.choice(len(X_te),ns,replace=False)
    scatter=[{"predicted":float(yp[i]),"actual":float(y_te.values[i])} for i in si]
    absr=np.abs(res); ti2=np.argsort(-absr)[:10]
    outs=[]
    for rk,idx in enumerate(ti2,1):
        li=float(levs[idx]); mi=tc*float(np.sqrt(s2*(1+li)))
        outs.append({"rank":rk,"index":int(X_te.index[idx]),"actual":float(y_te.values[idx]),
                     "predicted":float(yp[idx]),"residual":float(res[idx]),
                     "abs_error":float(absr[idx]),"pct_error":float(abs(res[idx])/max(abs(y_te.values[idx]),1e-6)*100),
                     "inside_pi":bool(absr[idx]<=mi)})
    return {"k":k,"features":list(cols),"rmse":rmse,"r2":r2_v,"adj_r2":adj_r2,"mae":mae,"f_stat":f_s,
            "cv_rmse":cv_rmse,"cv_r2":cv_r2,"sigma2":s2,"t_critical":tc,"mean_pi_width":mpiw,
            "pi_coverage":pi_cov,"inv_XtX":iXtX.tolist(),"coefficients":coefs,"outliers":outs,"test_scatter_sample":scatter}

MODEL_DATA={}
for mname, subsets in method_feature_subsets.items():
    print(f"  {mname}...")
    results=[]
    for ki,fs in enumerate(subsets):
        k=ki+1
        r=eval_ols(X_train,y_train,X_test,y_test,fs,len(X_train))
        r["k"]=k; results.append(r)
        if k<=2 or k==p_total: print(f"    k={k}: RMSE={r['rmse']:.4f}, R2={r['r2']:.4f}")
    MODEL_DATA[mname]=results

all_cv=[(m,r["k"],r["cv_rmse"]) for m,rs in MODEL_DATA.items() for r in rs]
best=min(all_cv,key=lambda x:x[2])
sweet_k=best[1]
print(f"\nBest: {best[0]} k={best[1]} CV RMSE={best[2]:.4f}")

feat_ranges={
    "age":{"name":"Age","min":36,"max":85,"step":1,"default":65,"unit":"yr"},
    "sex":{"name":"Sex (0=Male,1=Female)","min":0,"max":1,"step":1,"default":0,"unit":""},
    "test_time":{"name":"Test Time (days)","min":0,"max":215,"step":1,"default":100,"unit":"d"},
    "Jitter(%)":{"name":"Jitter (%)","min":0.001,"max":0.05,"step":0.001,"default":0.006,"unit":"%"},
    "Jitter(Abs)":{"name":"Jitter Abs","min":0.000003,"max":0.0003,"step":0.00001,"default":0.00004,"unit":"s"},
    "Jitter:RAP":{"name":"Jitter RAP","min":0.001,"max":0.03,"step":0.001,"default":0.003,"unit":""},
    "Jitter:PPQ5":{"name":"Jitter PPQ5","min":0.001,"max":0.03,"step":0.001,"default":0.003,"unit":""},
    "Jitter:DDP":{"name":"Jitter DDP","min":0.001,"max":0.08,"step":0.001,"default":0.01,"unit":""},
    "Shimmer":{"name":"Shimmer","min":0.003,"max":0.17,"step":0.001,"default":0.03,"unit":""},
    "Shimmer(dB)":{"name":"Shimmer (dB)","min":0.03,"max":1.6,"step":0.01,"default":0.28,"unit":"dB"},
    "Shimmer:APQ3":{"name":"Shimmer APQ3","min":0.001,"max":0.09,"step":0.001,"default":0.016,"unit":""},
    "Shimmer:APQ5":{"name":"Shimmer APQ5","min":0.001,"max":0.1,"step":0.001,"default":0.02,"unit":""},
    "Shimmer:APQ11":{"name":"Shimmer APQ11","min":0.002,"max":0.14,"step":0.001,"default":0.027,"unit":""},
    "Shimmer:DDA":{"name":"Shimmer DDA","min":0.003,"max":0.27,"step":0.001,"default":0.048,"unit":""},
    "NHR":{"name":"NHR","min":0.0003,"max":0.4,"step":0.001,"default":0.025,"unit":""},
    "HNR":{"name":"HNR (dB)","min":1.65,"max":33.0,"step":0.1,"default":21.7,"unit":"dB"},
    "RPDE":{"name":"RPDE","min":0.15,"max":0.97,"step":0.01,"default":0.54,"unit":""},
    "DFA":{"name":"DFA","min":0.51,"max":0.87,"step":0.01,"default":0.65,"unit":""},
    "PPE":{"name":"PPE","min":0.02,"max":0.73,"step":0.01,"default":0.22,"unit":""},
}
pred_meta=[]
for feat in feature_names:
    m=feat_ranges.get(feat,{})
    fc=feat.replace("(","").replace(")","").replace("%","Pct").replace(":","_").replace(".","_")
    pred_meta.append({"id":f"numPark{fc}","sliderId":f"rangePark{fc}","labelId":f"labelPark{fc}",
                      "name":m.get("name",feat),"feature":feat,
                      "min":m.get("min",float(X_raw[feat].min())),"max":m.get("max",float(X_raw[feat].max())),
                      "step":m.get("step",0.01),"default":m.get("default",float(X_raw[feat].median())),
                      "prefix":"","unit":m.get("unit","")})

ds_obj={
    "key":"parkinsons",
    "name":"Parkinsons Telemonitoring (UCI)",
    "target_name":"Motor UPDRS Score",
    "target_unit":"pts",
    "description":f"UCI Parkinsons Telemonitoring. Voice biomarkers predict motor UPDRS. {p_total} features (>13), 5875 samples. Best: k={sweet_k} ({best[0]}).",
    "badge_optimal":f"k = {sweet_k} Best-Case Sweet Spot ({best[0]})",
    "badge_overfitting":f"k = {p_total} Full Model",
    "k_labels":[str(k) for k in range(1,p_total+1)],
    "default_k":sweet_k,"sweet_spot_k":sweet_k,"overfitting_k":p_total,
    "features":feature_names,"predictor_features":pred_meta,"model_data":MODEL_DATA,
    "test_scatter_sample":MODEL_DATA[best[0]][sweet_k-1]["test_scatter_sample"],
    "final_model_comparison":[{"method":m,"test_rmse":MODEL_DATA[m][sweet_k-1]["rmse"],"test_r2":MODEL_DATA[m][sweet_k-1]["r2"]} for m in MODEL_DATA]
}

js_out=f'window.DATASETS["parkinsons"] = {json.dumps(ds_obj, indent=2, ensure_ascii=False)};\n'
with open("docs/parkinsons_data.js","w",encoding="utf-8") as f:
    f.write("// Parkinsons Telemonitoring Model Data (Auto-generated)\n")
    f.write(js_out)
print("Written: docs/parkinsons_data.js")

# Append to model_data.js
with open("docs/model_data.js","r",encoding="utf-8") as f:
    existing=f.read()
if '"parkinsons"' in existing:
    print("parkinsons already in model_data.js")
else:
    snippet=f',\n  "parkinsons": {json.dumps(ds_obj, indent=2, ensure_ascii=False)}'
    pos=existing.rfind("};")
    if pos!=-1:
        new=existing[:pos]+snippet+"\n};\n"
        with open("docs/model_data.js","w",encoding="utf-8") as f:
            f.write(new)
        print("Appended to docs/model_data.js")
    else:
        print("Could not find end of DATASETS")
print("Done!")
