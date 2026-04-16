# Time-Series Trend Discovery and Pattern Mining Using Data Mining Techniques

A complete data mining project for discovering market behavior patterns in financial time-series data, with both notebook-based analysis and an interactive Streamlit dashboard.

## Project Summary

This project analyzes stock time-series data (default: IBM) to identify trends, recurring motifs, volatility regimes, anomalies, and segment-level behavioral clusters. It combines statistical feature engineering with unsupervised learning and time-series similarity analysis.

The workflow includes:
- Data collection from Yahoo Finance or CSV
- Time-series preprocessing and feature engineering
- Trend/seasonal/residual decomposition
- Segmentation and clustering of market behavior windows
- DTW-based similarity and motif discovery
- Multi-method anomaly detection
- Interactive visualization and exploration in Streamlit

## Objectives

- Detect hidden structure in multi-year stock data
- Segment the time series into behaviorally meaningful windows
- Cluster similar market phases using engineered features
- Discover recurring patterns using Dynamic Time Warping (DTW)
- Detect suspicious or extreme events using statistical and model-based methods
- Present findings through a deployable dashboard

## Methodology

### 1. Data Ingestion

Supported sources:
- Local CSV file
- Uploaded CSV file
- Live Yahoo Finance fetch

Expected columns:
- Date, Time, Open, High, Low, Close, Volume

The parser also handles malformed exports (for example, an extra second ticker/header row).

### 2. Feature Engineering

Derived features include:
- Log_Return
- Daily_Return
- SMA_20
- EMA_20
- SMA_50
- Rolling_Vol (annualized)
- Momentum_10
- ROC_5
- BB_Upper
- BB_Lower
- HL_Range

### 3. Decomposition

STL-style decomposition is implemented using a Savitzky-Golay trend smoother and annual periodic seasonal extraction (period = 252 trading days), producing:
- Trend
- Seasonal
- Residual

### 4. Segmentation and Clustering

- Non-overlapping window segmentation
- Segment representation using feature means and standard deviations
- Standard scaling
- KMeans and Agglomerative clustering
- K selection using:
  - Inertia (Elbow)
  - Silhouette score
  - Davies-Bouldin index
- PCA projection for visual separability

### 5. DTW and Motif Discovery

- Pairwise DTW distance matrix on sampled normalized windows
- Motif discovery by finding minimum-DTW pairs per cluster

### 6. Anomaly Detection

Three complementary methods are combined:
- Z-score thresholding on returns
- IQR outlier detection on rolling volatility
- Isolation Forest on multivariate behavior features

Strong anomalies are defined by a configurable score threshold across methods.

## Interactive Dashboard

The Streamlit app provides seven analysis tabs:
1. Overview
2. Decomposition
3. Clustering
4. DTW and Motifs
5. Anomalies
6. Distribution
7. Cluster Heatmap

Main controls:
- Data source selection
- Ticker and years (Yahoo mode)
- Segment window size
- DTW sample size
- Cluster range
- Anomaly thresholds

## Repository Structure

- `DataMiningProject.ipynb`: Primary notebook pipeline
- `streamlit_app.py`: Interactive dashboard app
- `requirements.txt`: Python dependencies
- `IBM_5years_timeseries.csv`: Sample dataset
- `01_price_overview.png` to `12_cluster_heatmap.png`: Generated analysis figures

## Local Setup

1. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies

```powershell
pip install -r requirements.txt
```

3. Run dashboard

```powershell
streamlit run streamlit_app.py
```

4. Open in browser
- https://timeseriesanalysisdm.streamlit.app/

## Streamlit Community Cloud Deployment

1. Push this project to GitHub
2. Go to Streamlit Community Cloud
3. Create a new app and select this repository
4. Set:
- Branch: `main`
- Main file path: `streamlit_app.py`
5. Deploy

## Reproducibility Notes

- Most random processes use fixed seeds where applicable
- Results can vary slightly across package versions and data download timing

## Contributors

- Ammar Hassan
- Owais Imran
- Muhammad Ali saleem

## Acknowledgments

Built using:
- Streamlit
- Plotly
- pandas
- NumPy
- SciPy
- scikit-learn
- yfinance
