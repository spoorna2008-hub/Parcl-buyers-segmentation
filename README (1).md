# Parcl Buyer Segmentation Dashboard

Machine learning based buyer segmentation and investment profiling for real estate,
built for Parcl Co. in collaboration with Unified Mentor.

## What this does
Clusters 2,000 real estate buyers into four segments (Premium space buyers,
Office/commercial buyers, Standard residential buyers, High-volume portfolio
investors) using K-means clustering, validated against hierarchical clustering.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `app.py` - Streamlit dashboard (4 modules: segmentation overview, investor
  behavior, geographic analysis, segment insights)
- `buyer_profiles_final.csv` - cleaned, feature-engineered, clustered buyer data
- `Buyer_Segmentation_Research_Report.docx` - full methodology and findings

## Methodology summary
Data cleaning -> feature engineering (transaction aggregation per client) ->
encoding/scaling -> K-means (k=4, chosen via elbow + silhouette) validated with
hierarchical clustering -> cluster interpretation -> Streamlit dashboard.
