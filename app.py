import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Customer Segmentation & Predictive Analytics", page_icon="📊", layout="wide")
st.title("📊 Customer Segmentation & Predictive Analytics")
st.caption("Anonymized retail transactions • RFM segmentation • Optional K-means • Future spending estimate")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

@st.cache_data
def load_demo_data():
    customers = pd.read_csv(os.path.join(DATA_DIR, "customers.csv"))
    orders = pd.read_csv(os.path.join(DATA_DIR, "orders.csv"), parse_dates=["order_date"])
    items = pd.read_csv(os.path.join(DATA_DIR, "order_items.csv"))
    return customers, orders, items

def build_order_values(orders, items):
    merged = items.merge(orders, on="order_id", how="inner")
    merged["line_total"] = merged["quantity"] * merged["unit_price"]
    order_values = merged.groupby(["order_id", "customer_id", "order_date", "order_status"], as_index=False)["line_total"].sum()
    # Only completed orders count toward customer spend
    return order_values[order_values["order_status"].str.lower().eq("completed")].copy()

with st.sidebar:
    st.header("Data")
    uploaded_customers = st.file_uploader("customers.csv (optional)", type="csv", key="customers")
    uploaded_orders = st.file_uploader("orders.csv (optional)", type="csv", key="orders")
    uploaded_items = st.file_uploader("order_items.csv (optional)", type="csv", key="items")
    st.caption("Upload all three files together, or use the included demo dataset.")
    st.divider()
    st.header("Analysis settings")
    method = st.radio("Segmentation method", ["Rule-based RFM", "K-means clustering", "Compare both"])
    k = st.slider("K-means clusters", min_value=2, max_value=8, value=4)
    st.caption("K-means is exploratory; cluster labels are not objective truths.")

try:
    if uploaded_customers and uploaded_orders and uploaded_items:
        customers = pd.read_csv(uploaded_customers)
        orders = pd.read_csv(uploaded_orders, parse_dates=["order_date"])
        items = pd.read_csv(uploaded_items)
    else:
        customers, orders, items = load_demo_data()
except Exception as e:
    st.error(f"Could not load the data: {e}")
    st.stop()

required = {
    "customers": {"customer_id"},
    "orders": {"order_id", "customer_id", "order_date", "order_status"},
    "items": {"order_id", "quantity", "unit_price"},
}
for name, df in [("customers", customers), ("orders", orders), ("items", items)]:
    missing = required[name] - set(df.columns)
    if missing:
        st.error(f"{name} file is missing required columns: {', '.join(sorted(missing))}")
        st.stop()

orders["order_date"] = pd.to_datetime(orders["order_date"], errors="coerce")
items["quantity"] = pd.to_numeric(items["quantity"], errors="coerce")
items["unit_price"] = pd.to_numeric(items["unit_price"], errors="coerce")
orders = orders.dropna(subset=["order_id", "customer_id", "order_date", "order_status"])
items = items.dropna(subset=["order_id", "quantity", "unit_price"])
items = items[(items["quantity"] > 0) & (items["unit_price"] >= 0)]
orders = orders[orders["customer_id"].isin(customers["customer_id"])]

order_values = build_order_values(orders, items)
if order_values.empty:
    st.warning("No completed orders were found. Check the order_status values.")
    st.stop()

analysis_date = order_values["order_date"].max().normalize() + pd.Timedelta(days=1)
rfm = order_values.groupby("customer_id").agg(
    last_purchase=("order_date", "max"),
    
    frequency=("order_id", "nunique"),
    monetary=("line_total", "sum"),
).reset_index()
rfm["recency_days"] = (analysis_date - rfm["last_purchase"].dt.normalize()).dt.days
all_ids = customers[["customer_id"]].drop_duplicates()
rfm = all_ids.merge(rfm, on="customer_id", how="left")
rfm["frequency"] = rfm["frequency"].fillna(0).astype(int)
rfm["monetary"] = rfm["monetary"].fillna(0.0)
rfm["recency_days"] = rfm["recency_days"].fillna((analysis_date - order_values["order_date"].min().normalize()).days + 1)
rfm["last_purchase"] = rfm["last_purchase"].fillna(pd.NaT)
rfm["average_order_value"] = np.where(rfm["frequency"] > 0, rfm["monetary"] / rfm["frequency"], 0)

def rfm_label(row):
    if row["frequency"] >= 5 and row["recency_days"] <= 60 and row["monetary"] >= rfm["monetary"].quantile(0.70):
        return "Champions"
    if row["recency_days"] <= 90 and row["frequency"] >= 2:
        return "Loyal / Active"
    if row["recency_days"] > 180 and row["frequency"] >= 2:
        return "At Risk"
    if row["frequency"] == 1:
        return "One-time"
    if row["frequency"] == 0:
        return "No Completed Purchase"
    return "Regular"

rfm["rfm_segment"] = rfm.apply(rfm_label, axis=1)

# K-means features use log transforms and scaling to reduce skew.
features = rfm[["recency_days", "frequency", "monetary"]].copy()
features["frequency"] = np.log1p(features["frequency"])
features["monetary"] = np.log1p(features["monetary"])
scaler = StandardScaler()
scaled = scaler.fit_transform(features)
k_actual = min(k, len(rfm))
if k_actual >= 2:
    km = KMeans(n_clusters=k_actual, random_state=42, n_init=10)
    rfm["kmeans_cluster"] = km.fit_predict(scaled).astype(str)
    try:
        sil = silhouette_score(scaled, rfm["kmeans_cluster"].astype(int)) if len(set(rfm["kmeans_cluster"])) > 1 else np.nan
    except Exception:
        sil = np.nan
else:
    rfm["kmeans_cluster"] = "0"
    sil = np.nan

st.subheader("Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", f"{len(customers):,}")
c2.metric("Completed orders", f"{len(order_values):,}")
c3.metric("Total completed spend", f"₹{order_values['line_total'].sum():,.0f}")
c4.metric("Average customer spend", f"₹{rfm['monetary'].mean():,.0f}")
st.caption(f"Recency reference date: {analysis_date.date()} (the day after the latest completed order). Cancelled and returned orders are excluded.")

left, right = st.columns(2)
with left:
    st.subheader("RFM segment distribution")
    counts = rfm["rfm_segment"].value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots()
    counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Segment")
    ax.set_ylabel("Number of customers")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
with right:
    st.subheader("Spending by RFM segment")
    spend = rfm.groupby("rfm_segment")["monetary"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots()
    spend.plot(kind="bar", ax=ax)
    ax.set_xlabel("Segment")
    ax.set_ylabel("Average completed spend (₹)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.subheader("Segment profiles")
profile = rfm.groupby("rfm_segment").agg(
    customers=("customer_id", "count"),
    avg_recency_days=("recency_days", "mean"),
    avg_orders=("frequency", "mean"),
    avg_spend=("monetary", "mean"),
    median_spend=("monetary", "median"),
).round(2).reset_index()
st.dataframe(profile, use_container_width=True, hide_index=True)

if method in ["K-means clustering", "Compare both"]:
    st.subheader("K-means cluster comparison")
    st.write("K-means groups customers using scaled recency, frequency and spending. Numeric cluster IDs are arbitrary; interpret them using their profiles.")
    st.metric("Silhouette score", "Not available" if pd.isna(sil) else f"{sil:.3f}")
    km_profile = rfm.groupby("kmeans_cluster").agg(
        customers=("customer_id", "count"),
        avg_recency_days=("recency_days", "mean"),
        avg_orders=("frequency", "mean"),
        avg_spend=("monetary", "mean"),
        median_spend=("monetary", "median"),
    ).round(2).reset_index()
    st.dataframe(km_profile, use_container_width=True, hide_index=True)
    fig, ax = plt.subplots()
    rfm.boxplot(column="monetary", by="kmeans_cluster", ax=ax)
    ax.set_title("Spending distribution by K-means cluster")
    fig.suptitle("")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Completed spend (₹)")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.subheader("Customer-level RFM table")
show = rfm[["customer_id", "recency_days", "frequency", "monetary", "average_order_value", "rfm_segment"]].copy()
if method in ["K-means clustering", "Compare both"]:
    show["kmeans_cluster"] = rfm["kmeans_cluster"]
st.dataframe(show.sort_values("monetary", ascending=False), use_container_width=True, hide_index=True)
st.download_button("Download RFM results (CSV)", show.to_csv(index=False).encode("utf-8"), "rfm_customer_segments.csv", "text/csv")

# Time-based spending prediction: use prior period as features and following 90 days as target.
st.subheader("Predictive analytics: estimate future 90-day spending")
st.write("This demo creates one training example per customer: historical RFM-like features before a cutoff predict spending in the following 90 days. It uses a time-based split, not a random split.")
latest_date = order_values["order_date"].max().normalize()
cutoff = latest_date - pd.Timedelta(days=90)
history = order_values[order_values["order_date"] <= cutoff]
future = order_values[(order_values["order_date"] > cutoff) & (order_values["order_date"] <= latest_date)]
if history.empty or future.empty:
    st.info("Not enough historical and future-period data for a meaningful spending model. Include transactions spanning at least 6 months.")
else:
    hist_features = history.groupby("customer_id").agg(
        hist_orders=("order_id", "nunique"),
        hist_spend=("line_total", "sum"),
        last_hist_purchase=("order_date", "max"),
    ).reset_index()
    hist_features["hist_recency_days"] = (cutoff - hist_features["last_hist_purchase"].dt.normalize()).dt.days
    hist_features = customers[["customer_id"]].merge(hist_features, on="customer_id", how="left")
    hist_features[["hist_orders", "hist_spend"]] = hist_features[["hist_orders", "hist_spend"]].fillna(0)
    hist_features["hist_recency_days"] = hist_features["hist_recency_days"].fillna(365)
    targets = future.groupby("customer_id")["line_total"].sum().rename("future_90d_spend")
    model_data = hist_features.merge(targets, on="customer_id", how="left")
    model_data["future_90d_spend"] = model_data["future_90d_spend"].fillna(0)
    X = model_data[["hist_orders", "hist_spend", "hist_recency_days"]]
    y = model_data["future_90d_spend"]
    if len(model_data) < 20 or y.nunique() < 2:
        st.info("There is not enough variation in the sample to train a useful model.")
    else:
        # A chronological customer-ID split is used only as a simple demo split; the target period remains later than features.
        split_at = int(len(model_data) * 0.8)
        X_train, X_test = X.iloc[:split_at], X.iloc[split_at:]
        y_train, y_test = y.iloc[:split_at], y.iloc[split_at:]
        model = RandomForestRegressor(n_estimators=150, random_state=42, min_samples_leaf=3)
        model.fit(X_train, y_train)
        pred = np.maximum(model.predict(X_test), 0)
        mae = mean_absolute_error(y_test, pred)
        rmse = mean_squared_error(y_test, pred) ** 0.5
        r2 = r2_score(y_test, pred) if y_test.nunique() > 1 else np.nan
        a, b, c = st.columns(3)
        a.metric("MAE", f"₹{mae:,.2f}")
        b.metric("RMSE", f"₹{rmse:,.2f}")
        c.metric("R²", "Not meaningful" if pd.isna(r2) else f"{r2:.3f}")
        model_data["predicted_future_90d_spend"] = np.maximum(model.predict(X), 0)
        st.dataframe(model_data[["customer_id", "hist_orders", "hist_spend", "hist_recency_days", "future_90d_spend", "predicted_future_90d_spend"]].round(2), use_container_width=True, hide_index=True)
        st.download_button("Download spending predictions (CSV)", model_data.to_csv(index=False).encode("utf-8"), "spending_predictions.csv", "text/csv")
        st.caption("Important: synthetic/demo data and a small sample can produce misleading metrics. Validate with real, representative data before business use.")

st.subheader("How to interpret the segments")
st.markdown("""
- **Champions:** recent, frequent, high-spending customers under the chosen rules.
- **Loyal / Active:** recent customers with repeat purchases.
- **At Risk:** customers with multiple purchases but a long time since their last purchase.
- **One-time:** exactly one completed order.
- **Regular:** customers who do not match the other rules.
- **No Completed Purchase:** customers without a completed order in the supplied data.

These are practical labels, not facts about customer motivation. Thresholds should be reviewed with business stakeholders.
""")
st.info("Privacy note: the included dataset is synthetic and uses anonymous IDs. Do not upload names, phone numbers, email addresses, payment details, or other direct identifiers.")
