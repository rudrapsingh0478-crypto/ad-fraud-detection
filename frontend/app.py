"""Streamlit frontend.   Run from the project folder:
    streamlit run frontend/app.py
"""
import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import inference as inf  # noqa: E402

st.set_page_config(page_title="Ad Fraud Detection", layout="wide")
st.title("AI-Based Fraudulent Advertisement Detection System")
st.caption("Relationship-aware features from IP, device, app and time. Batch click scoring.")

with st.sidebar:
    st.header("Settings")
    model = st.selectbox("Model", ["RandomForest", "LightGBM"])
    variant = st.radio(
        "Features", ["proposed", "baseline"],
        format_func=lambda v: "Proposed (raw + 14 relationship features)" if v == "proposed"
        else "Baseline (6 raw features only)")
    min_clicks = st.slider("Minimum clicks per IP to review", 2, 500, 50)


@st.cache_resource(show_spinner="Building features and scoring clicks ...")
def run_scoring(raw: bytes, model: str, variant: str):
    return inf.score(pd.read_csv(io.BytesIO(raw)), model, variant)


tab_score, tab_compare, tab_about = st.tabs(["Score clicks", "Model comparison", "About and limits"])

# ------------------------------------------------------------------ score
with tab_score:
    st.write("Upload a CSV with the columns **ip, app, device, os, channel, click_time** "
             "(an **is_attributed** column is optional and enables accuracy metrics).")
    up = st.file_uploader("Click log (CSV)", type=["csv"])
    sample = inf.ROOT / "data" / "sample_clicks.csv"
    if up is None and sample.exists():
        if st.button("Load demo file (data/sample_clicks.csv)"):
            st.session_state["use_sample"] = True
    if up is not None:
        st.session_state["use_sample"] = False

    raw = None
    if up is not None:
        raw = up.getvalue()
    elif st.session_state.get("use_sample"):
        raw = sample.read_bytes()

    if raw is None:
        st.info("Upload a file to start. To create a demo file from the Kaggle data, run "
                "`python src/make_sample.py` in the project folder.")
    else:
        try:
            scored = run_scoring(raw, model, variant)
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()
        except ValueError as e:
            st.error(str(e))
            st.stop()

        h = inf.headline(scored)
        ips = inf.ip_summary(scored)
        review = inf.ips_to_review(ips, min_clicks)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Clicks scored", f"{h['clicks']:,}")
        c2.metric("Unique IPs", f"{h['unique_ips']:,}")
        c3.metric("Likely genuine clicks", f"{h['likely_genuine']:,}", f"{h['likely_genuine_pct']}%")
        c4.metric("IPs to review", f"{len(review):,}")
        if "roc_auc" in h:
            st.caption(f"The file contains labels: ROC-AUC {h['roc_auc']}, PR-AUC {h['pr_auc']}.")
        st.info("In TalkingData-style logs most clicks never lead to an install, so **Low quality** "
                "is not proof of fraud. Use the IP table below to decide which sources to review first.")

        st.subheader("Clicks by hour")
        st.bar_chart(scored.groupby("hour").size().rename("clicks"))

        st.subheader("IPs to review")
        st.caption("Heavy IPs where no click is predicted genuine. Sorted by number of clicks.")
        st.dataframe(review.head(200))
        st.download_button("Download IP review list (CSV)", review.to_csv(index=False).encode(),
                           "ips_to_review.csv", "text/csv")

        st.subheader("Explain one click")
        pool = st.radio("Pick a click from", ["Most likely genuine", "Heaviest IP under review"],
                        horizontal=True)
        if pool == "Most likely genuine" or review.empty:
            cand = scored.nlargest(20, "p_install")
        else:
            cand = scored[scored["ip"] == review.iloc[0]["ip"]].head(20)
        row_id = st.selectbox("Row ID", cand["row_id"].tolist())
        sel = scored[scored["row_id"] == row_id]
        row = sel.iloc[0]
        st.write(f"Install probability **{row['p_install']:.4f}**, predicted: **{row['quality']}**")
        bundle = inf.load_bundle(model, variant)
        X = sel[bundle["features"]]
        left, right = st.columns(2)
        left.caption("Feature values for this click")
        left.dataframe(X.iloc[0].rename("value").to_frame())
        sv = inf.explain(bundle, X)
        right.caption("SHAP contribution (positive pushes toward genuine)")
        if sv is None:
            right.info("Install the `shap` package to see contributions.")
        else:
            contrib = sv.iloc[0].sort_values(key=abs, ascending=False).head(10)
            right.bar_chart(contrib)

        st.subheader("Scored clicks (first 500 rows)")
        show = ["row_id", "ip", "app", "device", "os", "channel", "click_time", "p_install", "quality"]
        st.dataframe(scored[show].head(500))

# --------------------------------------------------------------- compare
with tab_compare:
    res = inf.load_results()
    if res is None:
        st.info("results/results.csv not found. Run train.py first.")
    else:
        order = [s for s in ["baseline (raw)", "+counts", "+diversity", "+timing", "+ratios",
                             "proposed (all)"] if s in set(res["feature_set"])]
        st.subheader("PR-AUC by feature set (higher is better)")
        st.bar_chart(res.pivot(index="feature_set", columns="model", values="pr_auc").loc[order])
        st.subheader("Cost of the extra features")
        st.dataframe(res[res["feature_set"].isin(["baseline (raw)", "proposed (all)"])]
                     [["model", "feature_set", "n_features", "train_s", "infer_ms_per_1000"]])
        st.subheader("All results")
        st.dataframe(res)
        shap_png = inf.ROOT / "results" / "shap_summary.png"
        if shap_png.exists():
            st.subheader("SHAP summary (Random Forest, proposed features)")
            st.image(str(shap_png))

# ----------------------------------------------------------------- about
with tab_about:
    st.markdown("""
**Feature groups (14 relationship-aware features)**
- Counts: clicks per IP, IP+app, IP+device+OS, IP+app+hour
- Diversity: unique apps, devices, channels and operating systems per IP
- Timing: seconds since the previous click from the same IP, IP+app, IP+device+OS
- Ratios: clicks per app, clicks per channel, share of an IP's clicks going to one app

**Limits to state honestly**
- The model predicts whether a click leads to an install. This is a proxy for click quality, not a confirmed fraud label.
- Features are computed over the uploaded batch. Use files of similar size and time span as the training data (about 1 million clicks); very small or very large files shift the count features.
- Single dataset (TalkingData). Batch scoring only, no real-time deployment.
""")
