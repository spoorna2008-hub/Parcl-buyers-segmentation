import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Parcl buyer intelligence", layout="wide")

SEGMENT_COLORS = {
    "Premium space buyers": "#378ADD",
    "Office/commercial buyers": "#1D9E75",
    "Standard residential buyers": "#D85A30",
    "High-volume portfolio investors": "#D4537E",
}

SEGMENT_BLURB = {
    "Premium space buyers": "Larger units, higher average price, mixed home/investment intent.",
    "Office/commercial buyers": "Own at least one office unit; steady mid-size investment.",
    "Standard residential buyers": "Smallest unit size and price, apartment-only, price sensitive.",
    "High-volume portfolio investors": "Multiple properties, highest total spend and satisfaction.",
}


@st.cache_data
def load_data():
    df = pd.read_csv("buyer_profiles_final.csv")
    return df


df = load_data()

st.title("Buyer segmentation and investment profiling")
st.caption("Machine learning based buyer segmentation for Parcl - real estate market intelligence")

# ---------------- Filters (sidebar) ----------------
st.sidebar.header("Filters")
countries = st.sidebar.multiselect("Country", sorted(df.country.unique()), default=[])
regions = st.sidebar.multiselect("Region", sorted(df.region.unique()), default=[])
purposes = st.sidebar.multiselect("Acquisition purpose", sorted(df.acquisition_purpose.unique()), default=[])
client_types = st.sidebar.multiselect("Client type", sorted(df.client_type.unique()), default=[])

filtered = df.copy()
if countries:
    filtered = filtered[filtered.country.isin(countries)]
if regions:
    filtered = filtered[filtered.region.isin(regions)]
if purposes:
    filtered = filtered[filtered.acquisition_purpose.isin(purposes)]
if client_types:
    filtered = filtered[filtered.client_type.isin(client_types)]

st.sidebar.metric("Buyers matching filters", len(filtered))

tab1, tab2, tab3, tab4 = st.tabs([
    "Buyer segmentation overview",
    "Investor behavior dashboard",
    "Geographic buyer analysis",
    "Segment insights panel",
])

# ---------------- Module 1: Buyer segmentation overview ----------------
with tab1:
    st.subheader("Cluster distribution")
    col1, col2 = st.columns([2, 1])
    with col1:
        counts = filtered.segment.value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(counts.index, counts.values, color=[SEGMENT_COLORS[s] for s in counts.index])
        ax.set_ylabel("Number of buyers")
        plt.xticks(rotation=20, ha="right")
        st.pyplot(fig)
    with col2:
        for seg in SEGMENT_COLORS:
            n = int((filtered.segment == seg).sum())
            st.metric(seg, n)

    st.subheader("Segment definitions")
    for seg, blurb in SEGMENT_BLURB.items():
        st.markdown(f"**{seg}** &nbsp;&mdash;&nbsp; {blurb}")

# ---------------- Module 2: Investor behavior dashboard ----------------
with tab2:
    st.subheader("Investment patterns by cluster")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for seg, c in SEGMENT_COLORS.items():
            sub = filtered[filtered.segment == seg]
            ax.scatter(sub.num_properties, sub.total_investment / 1e6, s=14, alpha=0.6, color=c, label=seg)
        ax.set_xlabel("Number of properties owned")
        ax.set_ylabel("Total investment ($M)")
        ax.legend(fontsize=7)
        st.pyplot(fig)
    with col2:
        st.markdown("**Average metrics by segment**")
        summary = filtered.groupby("segment")[
            ["num_properties", "total_investment", "avg_price", "price_per_sqft", "satisfaction_score"]
        ].mean().round(1)
        st.dataframe(summary)

    st.markdown("**Financing behavior**")
    loan_rate = filtered.groupby("segment")["loan_applied_flag"].mean().round(2) * 100
    st.bar_chart(loan_rate)

# ---------------- Module 3: Geographic buyer analysis ----------------
with tab3:
    st.subheader("Buyers by country and segment")
    top_countries = filtered.country.value_counts().nlargest(8).index
    geo = filtered[filtered.country.isin(top_countries)].groupby(["country", "segment"]).size().unstack(fill_value=0)
    if not geo.empty:
        geo = geo.loc[[c for c in top_countries if c in geo.index]]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        geo.plot(kind="bar", stacked=True, ax=ax, color=[SEGMENT_COLORS.get(c, "#888") for c in geo.columns])
        ax.set_ylabel("Number of buyers")
        plt.xticks(rotation=20, ha="right")
        ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
        st.pyplot(fig)
    else:
        st.info("No data for the current filter selection.")

    st.subheader("Country summary table")
    st.dataframe(filtered.groupby("country").agg(
        buyers=("client_id", "count"),
        avg_investment=("total_investment", "mean"),
    ).round(0).sort_values("buyers", ascending=False))

# ---------------- Module 4: Segment insights panel ----------------
with tab4:
    st.subheader("Descriptive statistics per segment")
    chosen = st.selectbox("Choose a segment", list(SEGMENT_COLORS.keys()))
    sub = filtered[filtered.segment == chosen]
    st.markdown(f"**{chosen}** &mdash; {SEGMENT_BLURB[chosen]}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Buyers", len(sub))
    c2.metric("Avg age", round(sub.age.mean(), 1) if len(sub) else "-")
    c3.metric("Avg investment", f"${sub.total_investment.mean():,.0f}" if len(sub) else "-")
    c4.metric("Avg satisfaction", round(sub.satisfaction_score.mean(), 1) if len(sub) else "-")

    st.markdown("**Acquisition purpose split**")
    st.bar_chart(sub.acquisition_purpose.value_counts())

    st.markdown("**Referral channel split**")
    st.bar_chart(sub.referral_channel.value_counts())
