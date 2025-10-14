import io
import os
from datetime import timedelta, date
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dateutil.relativedelta import relativedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import streamlit as st

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="E-commerce Growth Command Center",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Data Ingress Options (priority order)
# 1) Sidebar GitHub/HTTP URL (raw .xlsx or .csv)
# 2) Local file in repo: ecommerce_clickstream_transactions.xlsx
# 3) Streamlit uploader (.xlsx or .csv)
# =============================================================================

DEFAULT_LOCAL_XLSX = "ecommerce_clickstream_transactions.xlsx"
DEFAULT_SHEET = "ecommerce_clickstream_transacti"  # per spec (auto-detect if different)

# ------------------------------
# Definitions quick reference
# ------------------------------
def definitions_ui():
    content = '''
**Definitions & Formulas**

- **Total Revenue** = Σ Amount over **purchase** events within date range (ignoring NaN; excluding negative).
- **AOV (Average Order Value)** = Σ Amount over purchase / count(purchase events).
- **Conversion (Session → Purchase)** = unique sessions with ≥1 purchase / total unique sessions.
- **Repeat Purchase Rate** = users with ≥2 purchases / users with ≥1 purchase.
- **Active Users (7d / 30d)** = unique users with any event in last 7/30 days within selected range.
- **ARPU (Average Revenue per User)** = Σ Amount over purchase / unique users with any event.

- **Retention (cohort)** = active users in offset-month / cohort size.
- **CLV (finite horizon)** ≈ Σ_{m=1..H} (Expected Monthly Spend × Expected Retention^m) / (1 + r_m)^m,
  where H ∈ {6,12} and r_m is monthly discount rate derived from annual rate.
- **Churn Probability (30d)** = Logistic model on features {recency_days, frequency, monetary, avg_ticket},
  labeling whether user purchases in the next 30 days within the dataset window.
    '''
    try:
        pop = st.sidebar.popover("Definitions & Formulas")
        with pop:
            st.markdown(content)
    except Exception:
        with st.sidebar.expander("Definitions & Formulas", expanded=False):
            st.markdown(content)

definitions_ui()

# ------------------------------
# Robust column matching
# ------------------------------
COL_RULES = {
    "timestamp": ["timestamp", "time", "date", "datetime"],
    "user": ["userid", "user_id", "user"],
    "session": ["sessionid", "session_id", "session"],
    "event": ["eventtype", "event", "action", "activity", "step"],
    "product": ["productid", "product_id", "sku", "item"],
    "amount": ["amount", "price", "revenue", "value"],
}

def find_col(df: pd.DataFrame, keys: List[str]) -> str:
    for token in keys:
        for c in df.columns:
            if token in c.lower():
                return c
    return ""

def auto_map_columns(df: pd.DataFrame) -> Dict[str, str]:
    return {logical: find_col(df, tokens) for logical, tokens in COL_RULES.items()}

def column_mapping_helper(df: pd.DataFrame, mapping: Dict[str, str]) -> Dict[str, str]:
    st.info("If auto-detection is incorrect, adjust the mappings below and click **Apply**.")
    new_map = {}
    for logical in ["timestamp", "user", "session", "event", "product", "amount"]:
        options = [""] + list(df.columns)
        idx = options.index(mapping.get(logical, "")) if mapping.get(logical, "") in options else 0
        selection = st.selectbox(f"{logical} column", options, index=idx)
        new_map[logical] = selection
    return new_map

# ------------------------------
# Event normalization
# ------------------------------
def normalize_event(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.lower().fillna("")
    out = pd.Series(np.full(len(s), "", dtype=object), index=s.index)
    out = np.where(s.str.contains(r"(purchase|order|sale|success)"), "purchase", out)
    out = np.where(s.str.contains(r"(checkout|payment|pay|address|shipping)"), "checkout", out)
    out = np.where(s.str.contains(r"(add|cart|atc)"), "add_to_cart", out)
    out = np.where(s.str.contains(r"(view|product_view|detail|browse)"), "view", out)
    return pd.Series(out, index=s.index).astype(str)

# ------------------------------
# IO helpers (cached)
# ------------------------------
@st.cache_data(show_spinner=True)
def _read_excel_bytes(file_bytes: bytes, sheet: str | None) -> pd.DataFrame:
    # Require openpyxl for .xlsx
    try:
        import openpyxl  # noqa: F401  # ensure dependency is present
    except Exception as e:
        raise ImportError(
            "Reading .xlsx requires 'openpyxl'. "
            "Install it with: pip install openpyxl"
        ) from e
    bio = io.BytesIO(file_bytes)
    # If sheet specified, use it; else let pandas default to first sheet
    if sheet:
        return pd.read_excel(bio, sheet_name=sheet)
    return pd.read_excel(bio)

@st.cache_data(show_spinner=True)
def _read_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))

@st.cache_data(show_spinner=True)
def _http_get(url: str) -> bytes:
    # Minimal stdlib HTTP getter (no extra deps)
    import urllib.request
    with urllib.request.urlopen(url) as resp:
        return resp.read()

# ------------------------------
# Preprocess (cached)
# ------------------------------
@st.cache_data(show_spinner=True)
def preprocess(df: pd.DataFrame, mapping: Dict[str, str]) -> Tuple[pd.DataFrame, Dict[str, str], pd.Timestamp, pd.Timestamp]:
    required = ["timestamp", "user", "session", "event", "product", "amount"]
    missing = [r for r in required if not mapping.get(r)]
    if missing:
        raise ValueError(f"Missing mapped columns for: {', '.join(missing)}")

    rename_map = {mapping[k]: k for k in required if mapping.get(k)}
    df = df.rename(columns=rename_map)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=False)
    df = df.dropna(subset=["timestamp"])
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["event_norm"] = normalize_event(df["event"])

    min_dt = pd.to_datetime(df["timestamp"].min()).normalize()
    max_dt = pd.to_datetime(df["timestamp"].max()).normalize()

    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour

    return df, {"canonical": list(df.columns)}, pd.Timestamp(min_dt), pd.Timestamp(max_dt)

def filter_by_date(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    return df.loc[(df["timestamp"].dt.date >= start) & (df["timestamp"].dt.date <= end)].copy()

def purchase_df(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["event_norm"] == "purchase") & (df["amount"].notna()) & (df["amount"] >= 0)]

# ------------------------------
# KPIs
# ------------------------------
def compute_kpis(df: pd.DataFrame, start: date, end: date, full_df: pd.DataFrame) -> Dict[str, float]:
    dff = filter_by_date(df, start, end)
    purch = purchase_df(dff)

    total_rev = float(purch["amount"].sum()) if not purch.empty else 0.0
    aov = float(purch["amount"].mean()) if not purch.empty else 0.0

    ses_total = dff["session"].nunique()
    ses_with_purchase = dff.loc[dff["event_norm"] == "purchase", "session"].nunique()
    conversion = (ses_with_purchase / ses_total) if ses_total > 0 else 0.0

    users_with_1p = dff.loc[dff["event_norm"] == "purchase", "user"].nunique()
    users_with_2p = dff.loc[dff["event_norm"] == "purchase"].groupby("user").size()
    users_with_2p = int((users_with_2p >= 2).sum())
    repeat_rate = (users_with_2p / users_with_1p) if users_with_1p > 0 else 0.0

    end_dt = pd.to_datetime(end)
    au7 = filter_by_date(dff, max(start, (end_dt - timedelta(days=6)).date()), end)["user"].nunique()
    au30 = filter_by_date(dff, max(start, (end_dt - timedelta(days=29)).date()), end)["user"].nunique()

    users_active = dff["user"].nunique()
    arpu = (total_rev / users_active) if users_active > 0 else 0.0

    def delta_over(days: int) -> float | None:
        prev_start = (pd.to_datetime(start) - timedelta(days=days)).date()
        prev_end = (pd.to_datetime(end) - timedelta(days=days)).date()
        prev = filter_by_date(full_df, prev_start, prev_end)
        prev_p = purchase_df(prev)
        if prev_p.empty:
            return None
        prev_rev = float(prev_p["amount"].sum())
        if prev_rev == 0:
            return None
        return (total_rev - prev_rev) / prev_rev

    d30 = delta_over(30)
    d90 = delta_over(90)

    try:
        prev_start_y = (pd.to_datetime(start) - relativedelta(years=1)).date()
        prev_end_y = (pd.to_datetime(end) - relativedelta(years=1)).date()
        prev_y = filter_by_date(full_df, prev_start_y, prev_end_y)
        prev_y_p = purchase_df(prev_y)
        yoy = None
        if not prev_y_p.empty:
            prev_y_rev = float(prev_y_p["amount"].sum())
            if prev_y_rev != 0:
                yoy = (total_rev - prev_y_rev) / prev_y_rev
    except Exception:
        yoy = None

    return {
        "total_revenue": total_rev,
        "aov": aov,
        "conversion": conversion,
        "repeat_rate": repeat_rate,
        "active_users_7": int(au7),
        "active_users_30": int(au30),
        "arpu": arpu,
        "delta_30": d30,
        "delta_90": d90,
        "yoy": yoy,
    }

# ------------------------------
# Charts
# ------------------------------
def daily_revenue_chart(df: pd.DataFrame, start: date, end: date):
    dff = filter_by_date(df, start, end)
    purch = purchase_df(dff)
    if purch.empty:
        st.info("No purchase events in selected range.")
        return
    g = purch.groupby("date")["amount"].sum().reset_index().sort_values("date")
    g["ma7"] = g["amount"].rolling(7, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=g["date"], y=g["amount"], name="Daily Revenue"))
    fig.add_trace(go.Scatter(x=g["date"], y=g["ma7"], mode="lines", name="7-day MA"))
    fig.update_layout(title="Daily Revenue (with 7-day MA)", xaxis_title="Date", yaxis_title="Revenue")
    st.plotly_chart(fig, use_container_width=True)

def new_vs_returning(df: pd.DataFrame, start: date, end: date):
    dff = filter_by_date(df, start, end)
    first_event = df.groupby("user")["date"].min()
    dff = dff.assign(first_date=dff["user"].map(first_event))
    daily = dff.groupby(["date"]).apply(
        lambda x: pd.Series({
            "new": x.loc[x["date"] == x["first_date"], "user"].nunique(),
            "returning": x.loc[x["date"] != x["first_date"], "user"].nunique()
        })
    ).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=daily["date"], y=daily["new"], name="New Users"))
    fig.add_trace(go.Bar(x=daily["date"], y=daily["returning"], name="Returning Users"))
    fig.update_layout(barmode="stack", title="New vs. Returning Users", xaxis_title="Date", yaxis_title="Users")
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# Funnel & Drop-off
# ------------------------------
def session_stage_reach(df: pd.DataFrame) -> pd.DataFrame:
    piv = df.pivot_table(index="session", columns="event_norm", values="timestamp", aggfunc="min")
    for s in ["view", "add_to_cart", "checkout", "purchase"]:
        if s not in piv.columns:
            piv[s] = pd.NaT
    piv = piv.reset_index()
    for s in ["view", "add_to_cart", "checkout", "purchase"]:
        piv[f"has_{s}"] = piv[s].notna()
    return piv

def horizontal_funnel(df: pd.DataFrame):
    if df.empty:
        st.info("No sessions in selected range.")
        return
    order = ["view", "add_to_cart", "checkout", "purchase"]
    counts = []
    for i, s in enumerate(order):
        if i == 0:
            n = int(df[f"has_{s}"].sum())
        else:
            prev = order[i-1]
            n = int(((df[f"has_{prev}"]) & (df[f"has_{s}"])).sum())
        counts.append((s, n))
    rates = []
    for i in range(1, len(counts)):
        prev = counts[i-1][1]
        cur = counts[i][1]
        rate = (cur / prev) if prev > 0 else 0.0
        rates.append(rate)
    stages = [c[0].replace("_", " ").title() for c in counts]
    values = [c[1] for c in counts]
    fig = go.Figure(go.Bar(x=values, y=stages, orientation="h", text=values, textposition="auto"))
    fig.update_layout(title="Session Funnel (Unique Sessions Reaching Each Stage)", xaxis_title="Sessions", yaxis_title="Stage")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Step-through rates: " + " → ".join([f"{int(r*100)}%" for r in rates]))

def median_time_between(df_events: pd.DataFrame, start: date, end: date):
    dff = filter_by_date(df_events, start, end)
    sgrp = dff.groupby(["session", "event_norm"])["timestamp"].min().unstack()
    medians = {}
    pairs = [("view", "add_to_cart"), ("add_to_cart", "checkout"), ("checkout", "purchase"), ("view", "purchase")]
    for a, b in pairs:
        if (a in sgrp.columns) and (b in sgrp.columns):
            delta = (sgrp[b] - sgrp[a]).dropna().dt.total_seconds() / 60.0
            if len(delta) > 0:
                medians[f"{a}→{b}"] = float(np.median(delta))
    if medians:
        dfm = pd.DataFrame({"stage_pair": list(medians.keys()), "median_minutes": list(medians.values())})
        st.dataframe(dfm, use_container_width=True)
        st.download_button("Download median times (CSV)", dfm.to_csv(index=False).encode("utf-8"), "median_stage_times.csv", "text/csv")
    else:
        st.info("Not enough sessions with both stages to compute median times.")

def dropoff_analytics(dff: pd.DataFrame):
    order = ["view", "add_to_cart", "checkout", "purchase"]
    sreach = session_stage_reach(dff)
    last_stage = []
    for _, row in sreach.iterrows():
        stage = None
        for s in reversed(order):
            if row[f"has_{s}"]:
                stage = s
                break
        last_stage.append(stage or "none")
    sreach["last_stage"] = last_stage
    sess_info = dff.groupby("session").agg(first_ts=("timestamp", "min"), hour=("hour", "median"))
    top_products = dff.loc[dff["event_norm"] == "purchase"].groupby("product")["amount"].sum().sort_values(ascending=False).head(10).index.tolist()
    dff["product_bucket"] = np.where(dff["product"].isin(top_products), dff["product"], "Other")
    prod_mode = dff.groupby("session")["product_bucket"].agg(lambda x: x.mode().iat[0] if len(x.mode()) > 0 else "Other")
    sreach = sreach.join(sess_info).join(prod_mode.rename("product_bucket"))
    agg = sreach.groupby(["last_stage", "hour", "product_bucket"]).size().reset_index(name="sessions")
    if agg.empty:
        st.info("No drop-off diagnostics available in selected range.")
        return
    fig = px.bar(agg, x="sessions", y="last_stage", color="product_bucket", facet_col="hour", orientation="h",
                 title="Top Drop-off Points by Hour and Product Bucket")
    st.plotly_chart(fig, use_container_width=True)
    stuck = sreach[sreach["last_stage"].isin(["view", "add_to_cart", "checkout"])].copy()
    st.dataframe(stuck[["session", "last_stage", "hour", "product_bucket"]].head(1000), use_container_width=True)
    st.download_button("Download stuck sessions (CSV)", stuck[["session", "last_stage", "hour", "product_bucket"]].to_csv(index=False).encode("utf-8"), "stuck_sessions.csv", "text/csv")

def leakage_cohort(dff: pd.DataFrame):
    sess_first = dff.sort_values(["session", "timestamp"]).groupby("session").first().reset_index()
    tbl = sess_first.groupby([pd.to_datetime(sess_first["timestamp"]).dt.date, "event_norm"]).size().reset_index(name="sessions")
    if tbl.empty:
        st.info("No leakage cohort data available.")
        return
    st.dataframe(tbl, use_container_width=True)
    st.download_button("Download leakage cohort (CSV)", tbl.to_csv(index=False).encode("utf-8"), "leakage_cohort.csv", "text/csv")

# ------------------------------
# Cohorts & Retention
# ------------------------------
def cohorts_tab(df: pd.DataFrame, start: date, end: date):
    dff = filter_by_date(df, start, end)
    purch = purchase_df(df)
    if purch.empty:
        st.info("No purchases to form cohorts.")
        return
    first_purchase = purch.groupby("user")["timestamp"].min()
    cohort_month = first_purchase.dt.to_period("M")
    cohort_df = pd.DataFrame({"user": first_purchase.index, "cohort": cohort_month.astype(str)})
    dff["month"] = pd.to_datetime(dff["date"]).astype("datetime64[M]")
    active_by_month = dff.groupby(["user", "month"]).size().reset_index(name="events")
    data = active_by_month.merge(cohort_df, on="user", how="left").dropna(subset=["cohort"])
    data["offset"] = ((pd.to_datetime(data["month"]).dt.year - pd.to_datetime(data["cohort"]).dt.year) * 12 +
                      (pd.to_datetime(data["month"]).dt.month - pd.to_datetime(data["cohort"]).dt.month))
    data = data[(data["offset"] >= 0) & (data["offset"] <= 6)]
    cohort_sizes = cohort_df.groupby("cohort").size().rename("size").reset_index()
    retained = data.groupby(["cohort", "offset"])["user"].nunique().reset_index(name="active_users")
    grid = retained.merge(cohort_sizes, on="cohort", how="left")
    grid["retention"] = grid["active_users"] / grid["size"]
    purch_dff = purchase_df(dff)
    purch_dff["month"] = pd.to_datetime(purch_dff["date"]).astype("datetime64[M]")
    purch_with_cohort = purch_dff.merge(cohort_df, on="user", how="left").dropna(subset=["cohort"])
    purch_with_cohort["offset"] = ((pd.to_datetime(purch_with_cohort["month"]).dt.year - pd.to_datetime(purch_with_cohort["cohort"]).dt.year) * 12 +
                                   (pd.to_datetime(purch_with_cohort["month"]).dt.month - pd.to_datetime(purch_with_cohort["cohort"]).dt.month))
    purch_with_cohort = purch_with_cohort[(purch_with_cohort["offset"] >= 0) & (purch_with_cohort["offset"] <= 6)]
    rev_grid = purch_with_cohort.groupby(["cohort", "offset"])["amount"].sum().reset_index(name="revenue")
    mode = st.radio("Heatmap values", ["Retention %", "Revenue"], horizontal=True)
    if mode == "Retention %":
        mat = grid.pivot(index="cohort", columns="offset", values="retention").fillna(0)
        title = "Cohort Retention (0–6 months)"
    else:
        mat = rev_grid.pivot(index="cohort", columns="offset", values="revenue").fillna(0.0)
        title = "Cohort Revenue (0–6 months)"
    fig = go.Figure(data=go.Heatmap(
        z=mat.values,
        x=mat.columns.astype(int).tolist(),
        y=mat.index.tolist(),
        coloraxis="coloraxis",
        hovertemplate="Cohort %{y}<br>Offset %{x}<br>Value %{z}"
    ))
    fig.update_layout(title=title, coloraxis_colorscale="Blues")
    st.plotly_chart(fig, use_container_width=True)
    purch_user = purch_dff.sort_values(["user", "timestamp"])
    purch_user["prev_ts"] = purch_user.groupby("user")["timestamp"].shift(1)
    gaps = (purch_user["timestamp"] - purch_user["prev_ts"]).dt.days.dropna()
    if len(gaps) > 0:
        fig2 = px.histogram(gaps, nbins=30, title="Inter-purchase Gap Distribution (days)")
        fig2.update_xaxes(title="Days")
        fig2.update_yaxes(title="Frequency")
        st.plotly_chart(fig2, use_container_width=True)
    clv = purch_with_cohort.groupby(["cohort", "offset"])["amount"].sum().reset_index()
    clv["cum_rev"] = clv.groupby("cohort")["amount"].cumsum()
    st.dataframe(clv.pivot(index="cohort", columns="offset", values="cum_rev").fillna(0), use_container_width=True)
    st.download_button("Download cohort CLV table (CSV)", clv.to_csv(index=False).encode("utf-8"), "cohort_clv.csv", "text/csv")

# ------------------------------
# RFM Segmentation
# ------------------------------
def rfm_segmentation(df: pd.DataFrame, start: date, end: date):
    dff = filter_by_date(df, start, end)
    purch = purchase_df(dff)
    if purch.empty:
        st.info("No purchases in selected range; RFM unavailable.")
        return
    ref_date = pd.to_datetime(end)
    g = purch.groupby("user").agg(
        last_purchase=("timestamp", "max"),
        freq=("timestamp", "count"),
        monetary=("amount", "sum")
    ).reset_index()
    g["recency_days"] = (ref_date - g["last_purchase"]).dt.days.clip(lower=0)
    feat = g[["recency_days", "freq", "monetary"]].copy()
    k = st.slider("Number of clusters (k)", min_value=4, max_value=6, value=5, step=1)
    pipe = Pipeline([
        ("log", FunctionTransformer(func=np.log1p, validate=False)),
        ("scale", StandardScaler()),
        ("kmeans", KMeans(n_clusters=k, n_init="auto", random_state=42))
    ])
    labels = pipe.fit_predict(feat)
    g["cluster"] = labels
    means = g.groupby("cluster")[["recency_days", "freq", "monetary"]].mean()
    rank_r = means["recency_days"].rank(ascending=False)
    rank_f = means["freq"].rank(ascending=True) * -1
    rank_m = means["monetary"].rank(ascending=True) * -1
    score = (rank_f + rank_m - rank_r)
    order = score.sort_values(ascending=False).index.tolist()
    labels_map = {}
    label_names = ["Champions", "Loyal", "Promising", "At-Risk", "Hibernating", "Hibernating+"]
    for i, cid in enumerate(order):
        labels_map[cid] = label_names[i] if i < len(label_names) else f"Segment {i+1}"
    g["segment"] = g["cluster"].map(labels_map)
    seg = g.groupby("segment").agg(users=("user", "nunique"), revenue=("monetary", "sum")).reset_index()
    seg["user_share"] = seg["users"] / seg["users"].sum()
    seg["rev_share"] = seg["revenue"] / seg["revenue"].sum()
    st.dataframe(seg.sort_values("rev_share", ascending=False), use_container_width=True)
    st.download_button("Download segment mix (CSV)", seg.to_csv(index=False).encode("utf-8"), "rfm_segment_mix.csv", "text/csv")
    aov = purch.groupby("user")["amount"].sum().mean()
    rep = purch.groupby("user").size()
    repeat_rate = (rep >= 2).sum() / rep.shape[0] if rep.shape[0] > 0 else 0.0
    st.caption(f"AOV (user-level mean spend): {aov:,.2f} | Overall repeat rate: {repeat_rate:.0%}")
    nba = {
        "Champions": "Early access, premium referrals, VIP perks.",
        "Loyal": "Tiered rewards, exclusive bundles, anniversary offers.",
        "Promising": "Welcome-back incentives, curated recommendations.",
        "At-Risk": "Win-back discounts, urgency messaging, service outreach.",
        "Hibernating": "Reactivation campaigns, loss-leader offers, reminder nudges."
    }
    tips = pd.DataFrame([{"segment": s, "next_best_action": nba.get(s, "Tailored outreach")} for s in seg["segment"]])
    st.dataframe(tips, use_container_width=True)
    st.download_button("Download NBA (CSV)", tips.to_csv(index=False).encode("utf-8"), "rfm_nba.csv", "text/csv")

# ------------------------------
# Product Performance & Pareto
# ------------------------------
def product_performance(df: pd.DataFrame, start: date, end: date):
    dff = filter_by_date(df, start, end)
    purch = purchase_df(dff)
    if purch.empty:
        st.info("No purchases in selected range.")
        return
    top = purch.groupby("product")["amount"].sum().reset_index().sort_values("amount", ascending=False)
    top["rank"] = np.arange(1, len(top) + 1)
    top["cum_share"] = top["amount"].cumsum() / top["amount"].sum()
    st.dataframe(top.head(50), use_container_width=True)
    st.download_button("Download top products (CSV)", top.to_csv(index=False).encode("utf-8"), "top_products.csv", "text/csv")
    st.plotly_chart(px.bar(top.head(20), x="product", y="amount", title="Top Products by Revenue").update_layout(xaxis_title="Product", yaxis_title="Revenue"), use_container_width=True)
    pareto = px.line(top, x="rank", y="cum_share", title="Pareto Curve (Cumulative Revenue Share)")
    pareto.add_hline(y=0.8, line_dash="dash")
    st.plotly_chart(pareto, use_container_width=True)
    days = max(1, (pd.to_datetime(end) - pd.to_datetime(start)).days + 1)
    velocity = purch.groupby("product").size().rename("purchases").reset_index()
    velocity["per_day"] = velocity["purchases"] / days
    stick = purch.groupby("product")["user"].nunique().rename("repeat_buyers").reset_index()
    prod_stats = velocity.merge(stick, on="product", how="outer").fillna(0)
    st.dataframe(prod_stats.sort_values("per_day", ascending=False).head(50), use_container_width=True)
    st.download_button("Download velocity & stickiness (CSV)", prod_stats.to_csv(index=False).encode("utf-8"), "product_velocity_stickiness.csv", "text/csv")
    views = dff[dff["event_norm"].isin(["view", "add_to_cart", "checkout"])]
    sess_views = views.groupby("session")["product"].apply(lambda x: list(pd.unique(x.dropna())))
    pair_counts = {}
    for plist in sess_views:
        for i in range(len(plist)):
            for j in range(i + 1, len(plist)):
                a, b = sorted([plist[i], plist[j]])
                pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1
    copurch = purch.groupby("session")["product"].apply(lambda x: list(pd.unique(x.dropna())))
    copair = set()
    for plist in copurch:
        for i in range(len(plist)):
            for j in range(i + 1, len(plist)):
                a, b = sorted([plist[i], plist[j]])
                copair.add((a, b))
    rows = []
    for (a, b), cnt in sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:200]:
        if (a, b) not in copair:
            rows.append({"pair": f"{a} | {b}", "co_views": cnt, "co_purchased": 0})
    cannib = pd.DataFrame(rows)
    if not cannib.empty:
        st.dataframe(cannib.head(50), use_container_width=True)
        st.download_button("Download cannibalization pairs (CSV)", cannib.to_csv(index=False).encode("utf-8"), "cannibalization_watch.csv", "text/csv")
    else:
        st.caption("No prominent co-viewed-but-not-co-purchased pairs detected.")

# ------------------------------
# Session Analytics
# ------------------------------
def session_analytics(df: pd.DataFrame, start: date, end: date):
    dff = filter_by_date(df, start, end)
    if dff.empty:
        st.info("No sessions in selected range.")
        return
    sess = dff.groupby("session").agg(
        events=("timestamp", "count"),
        duration_sec=("timestamp", lambda s: (s.max() - s.min()).total_seconds()),
    ).reset_index()
    sess["duration_min"] = (sess["duration_sec"] / 60.0).round(2)
    rev = purchase_df(dff).groupby("session")["amount"].sum().rename("revenue")
    sess = sess.join(rev, on="session").fillna({"revenue": 0.0})
    bounce = int(((sess["events"] == 1) & (sess["revenue"] == 0)).sum())
    st.caption(f"Bounce proxy (single-event sessions without purchase): {bounce:,}")
    st.plotly_chart(px.histogram(sess, x="duration_min", nbins=50, title="Session Duration (minutes)"), use_container_width=True)
    st.plotly_chart(px.histogram(sess, x="events", nbins=50, title="Events per Session"), use_container_width=True)
    flows = dff.sort_values(["session", "timestamp"]).groupby("session")["event_norm"].apply(list)
    edges = {}
    for seq in flows:
        if "purchase" in seq:
            for i in range(len(seq) - 1):
                a, b = seq[i], seq[i + 1]
                edges[(a, b)] = edges.get((a, b), 0) + 1
    if edges:
        top_edges = sorted(edges.items(), key=lambda kv: kv[1], reverse=True)[:12]
        nodes = sorted(list(set([e[0][0] for e in top_edges] + [e[0][1] for e in top_edges])))
        idx = {n: i for i, n in enumerate(nodes)}
        link = dict(
            source=[idx[e[0][0]] for e in top_edges],
            target=[idx[e[0][1]] for e in top_edges],
            value=[e[1] for e in top_edges]
        )
        sankey = go.Figure(data=[go.Sankey(node=dict(label=nodes, pad=15, thickness=20), link=link)])
        sankey.update_layout(title_text="Top Event Flows (ending in purchase)", font_size=12)
        st.plotly_chart(sankey, use_container_width=True)
    st.dataframe(sess.head(1000), use_container_width=True)
    st.download_button("Download session metrics (CSV)", sess.to_csv(index=False).encode("utf-8"), "session_metrics.csv", "text/csv")

# ------------------------------
# CLV-Lite & Risk
# ------------------------------
def clv_and_risk(df: pd.DataFrame, start: date, end: date):
    dff = filter_by_date(df, start, end)
    purch = purchase_df(dff)
    if purch.empty:
        st.info("No purchases in selected range.")
        return
    horizon = st.radio("CLV horizon (months)", [6, 12], horizontal=True, index=0)
    annual_disc = st.slider("Annual discount rate (%)", min_value=0, max_value=30, value=10, step=1)
    monthly_disc = (1 + annual_disc/100.0) ** (1/12.0) - 1
    ref_end = pd.to_datetime(end)
    recent_start = (ref_end - pd.Timedelta(days=89)).date()
    recent = filter_by_date(dff, max(start, recent_start), end)
    recent_purch = purchase_df(recent)
    monthly_spend = recent_purch.groupby("user")["amount"].sum() / max(1, ((ref_end.date() - max(start, recent_start)).days + 1) / 30.0)
    rp = recent_purch.groupby("user").size()
    retention = float(((rp >= 2).sum() / max(1, rp.shape[0]))) if rp.shape[0] > 0 else 0.3
    retention = float(np.clip(retention, 0.2, 0.95))
    users = purch["user"].unique()
    rows = []
    for u in users:
        m_spend = float(monthly_spend.get(u, recent_purch["amount"].mean() if not recent_purch.empty else 0.0))
        clv = 0.0
        for m in range(1, horizon + 1):
            disc = (1 + monthly_disc) ** m
            clv += (m_spend * (retention ** m)) / disc
        rows.append({"user": u, "est_monthly_spend": m_spend, "est_retention": retention, f"CLV_{horizon}m": clv})
    clv_df = pd.DataFrame(rows)
    feat = purch.groupby("user").agg(
        last_purchase=("timestamp", "max"),
        freq=("timestamp", "count"),
        monetary=("amount", "sum"),
        avg_ticket=("amount", "mean")
    ).reset_index()
    feat["recency_days"] = (ref_end - feat["last_purchase"]).dt.days.clip(lower=0)
    label_window_end = ref_end + pd.Timedelta(days=30)
    future = df[(df["timestamp"] > ref_end) & (df["timestamp"] <= label_window_end) & (df["event_norm"] == "purchase")]
    future_label = future.groupby("user").size().rename("p_next30").astype(int)
    feat = feat.join(future_label, on="user").fillna({"p_next30": 0})
    show_auc = False
    if feat["p_next30"].nunique() == 2 and len(feat) >= 30:
        X = feat[["recency_days", "freq", "monetary", "avg_ticket"]].fillna(0.0).values
        y = feat["p_next30"].values
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        probs = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, probs)
        show_auc = True
        feat["churn_prob_30d"] = 1 - probs
    else:
        feat["churn_prob_30d"] = 1 - 1 / (1 + np.exp(-( -0.01*feat["recency_days"] + 0.1*feat["freq"] + 0.0001*feat["monetary"])))
    out = feat.merge(clv_df, on="user", how="left")
    out["priority_score"] = out[f"CLV_{horizon}m"] * (1 - out["churn_prob_30d"])
    out = out.sort_values("priority_score", ascending=False).reset_index(drop=True)
    if show_auc:
        st.caption(f"Churn model ROC-AUC (in-sample, baseline): {auc:.3f}")
    st.dataframe(out[["user", f"CLV_{horizon}m", "churn_prob_30d", "priority_score"]].head(1000), use_container_width=True)
    st.download_button("Download prioritized retention list (CSV)", out.to_csv(index=False).encode("utf-8"), "retention_list.csv", "text/csv")

# ------------------------------
# Executive Overview
# ------------------------------
def executive_overview(df: pd.DataFrame, start: date, end: date, full_df: pd.DataFrame):
    k = compute_kpis(df, start, end, full_df)
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Total Revenue", f"{k['total_revenue']:,.2f}", delta=None)
    c2.metric("AOV", f"{k['aov']:,.2f}")
    c3.metric("Session→Purchase", f"{k['conversion']:.0%}")
    c4.metric("Repeat Purchase Rate", f"{k['repeat_rate']:.0%}")
    c5.metric("Active Users (7d)", f"{k['active_users_7']:,}")
    c6.metric("Active Users (30d)", f"{k['active_users_30']:,}")
    c7.metric("ARPU", f"{k['arpu']:,.2f}")
    d1, d2, d3 = st.columns(3)
    d1.metric("Δ 30-day Revenue", f"{k['delta_30']:.0%}" if k['delta_30'] is not None else "n/a")
    d2.metric("Δ 90-day Revenue", f"{k['delta_90']:.0%}" if k['delta_90'] is not None else "n/a")
    d3.metric("YoY Revenue", f"{k['yoy']:.0%}" if k['yoy'] is not None else "n/a")
    st.markdown("---")
    daily_revenue_chart(df, start, end)
    new_vs_returning(df, start, end)

# ------------------------------
# Sidebar Inputs & Data Loading
# ------------------------------
st.title("E-commerce Growth Command Center")

st.sidebar.markdown("### Data Source")
gh_url = st.sidebar.text_input("GitHub/HTTP raw URL (.xlsx or .csv)", value="", help="Paste the raw file URL. CSV requires no extra packages; XLSX requires 'openpyxl'.")
uploaded = st.sidebar.file_uploader("Or upload file (.xlsx or .csv)", type=["xlsx", "csv"])

df_loaded = None
sheet_name = None

# 1) URL path
if gh_url.strip():
    try:
        raw = _http_get(gh_url.strip())
        if gh_url.strip().lower().endswith(".csv"):
            df_loaded = _read_csv_bytes(raw)
            sheet_name = None
        else:
            # Try Excel
            # Build sheet selector from workbook if possible; otherwise fall back to default first sheet.
            try:
                import openpyxl  # noqa: F401
            except Exception as e:
                st.error("Reading .xlsx requires `openpyxl`. Install with: `pip install openpyxl` "
                         "or supply a CSV URL instead.")
                st.stop()
            # Probe sheets
            xls = pd.ExcelFile(io.BytesIO(raw))
            sheets = xls.sheet_names
            default_idx = sheets.index(DEFAULT_SHEET) if DEFAULT_SHEET in sheets else 0
            sheet_name = st.sidebar.selectbox("Sheet", sheets, index=default_idx, key="sheet_url")
            df_loaded = _read_excel_bytes(raw, sheet_name)
    except Exception as e:
        st.error(f"Failed to load from URL: {e}")
        st.stop()

# 2) Local default file
elif os.path.exists(DEFAULT_LOCAL_XLSX):
    try:
        # Probe sheets
        try:
            import openpyxl  # noqa: F401
        except Exception as e:
            st.error("Local .xlsx found but `openpyxl` is missing. Install with: `pip install openpyxl`, "
                     "or save/export the file as CSV and reload via URL/Uploader.")
            st.stop()
        xls = pd.ExcelFile(DEFAULT_LOCAL_XLSX)
        sheets = xls.sheet_names
        default_idx = sheets.index(DEFAULT_SHEET) if DEFAULT_SHEET in sheets else 0
        sheet_name = st.sidebar.selectbox("Sheet", sheets, index=default_idx, key="sheet_local")
        with open(DEFAULT_LOCAL_XLSX, "rb") as f:
            df_loaded = _read_excel_bytes(f.read(), sheet_name)
    except Exception as e:
        st.error(f"Failed to read local file '{DEFAULT_LOCAL_XLSX}': {e}")
        st.stop()

# 3) Streamlit uploader
elif uploaded is not None:
    try:
        raw = uploaded.getvalue()
        if uploaded.name.lower().endswith(".csv"):
            df_loaded = _read_csv_bytes(raw)
            sheet_name = None
        else:
            try:
                import openpyxl  # noqa: F401
            except Exception:
                st.error("Reading .xlsx requires `openpyxl`. Install with: `pip install openpyxl`, "
                         "or upload CSV instead.")
                st.stop()
            xls = pd.ExcelFile(io.BytesIO(raw))
            sheets = xls.sheet_names
            default_idx = sheets.index(DEFAULT_SHEET) if DEFAULT_SHEET in sheets else 0
            sheet_name = st.sidebar.selectbox("Sheet", sheets, index=default_idx, key="sheet_upload")
            df_loaded = _read_excel_bytes(raw, sheet_name)
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")
        st.stop()
else:
    st.info("Provide a GitHub/HTTP URL, or place 'ecommerce_clickstream_transactions.xlsx' in project root, or upload a file.")
    st.stop()

# Column mapping & preprocess
auto_map = auto_map_columns(df_loaded)
with st.sidebar.expander("Column Mapping", expanded=False):
    if st.checkbox("Adjust mappings", value=False):
        auto_map = column_mapping_helper(df_loaded, auto_map)

try:
    df, meta, min_dt, max_dt = preprocess(df_loaded, auto_map)
    min_date, max_date = min_dt.date(), max_dt.date()
except ValueError as e:
    st.error(str(e))
    st.stop()

# Date range
start_date, end_date = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)
if isinstance(start_date, (list, tuple)):
    start_date, end_date = start_date[0], start_date[1]

# ------------------------------
# Tabs
# ------------------------------
tabs = st.tabs([
    "Executive Overview",
    "Funnel & Drop-off",
    "Cohorts & Retention",
    "RFM Segmentation",
    "Product Performance & Pareto",
    "Session Analytics",
    "CLV-Lite & Risk",
])

with tabs[0]:
    executive_overview(df, start_date, end_date, df)

with tabs[1]:
    dff = filter_by_date(df, start_date, end_date)
    sreach = session_stage_reach(dff)
    horizontal_funnel(sreach)
    st.markdown("#### Median Time Between Stages")
    median_time_between(df, start_date, end_date)
    st.markdown("#### Drop-off & Stuck Sessions")
    dropoff_analytics(dff)
    st.markdown("#### Leakage Diagnostics (First Stage per Session by Day)")
    leakage_cohort(dff)

with tabs[2]:
    cohorts_tab(df, start_date, end_date)

with tabs[3]:
    rfm_segmentation(df, start_date, end_date)

with tabs[4]:
    product_performance(df, start_date, end_date)

with tabs[5]:
    session_analytics(df, start_date, end_date)

with tabs[6]:
    clv_and_risk(df, start_date, end_date)
