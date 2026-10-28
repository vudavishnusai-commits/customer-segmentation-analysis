# Customer Segmentation and Predictive Analytics System

A beginner-friendly Streamlit project for RFM analysis, optional K-means clustering, and a demonstration model that estimates future 90-day customer spending.

## Features
- Uses anonymized synthetic customer and transaction data.
- Calculates recency, frequency, monetary value and average order value.
- Applies interpretable rule-based RFM segments.
- Optionally runs K-means on log-transformed, standardized features.
- Shows segment size and spending profiles.
- Trains a Random Forest regressor to estimate next-90-day spend from historical transactions.
- Downloads customer segment and prediction tables as CSV.
- Includes a MySQL schema and a project report.

## Requirements
Python 3.10 or newer is recommended. Internet is needed only to install packages.

## Setup on Windows
1. Extract the ZIP file.
2. Open the project folder in VS Code.
3. Open Terminal > New Terminal.
4. Create a virtual environment:
   ```powershell
   py -m venv .venv
   ```
5. Activate it:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks activation, use Command Prompt and run:
   ```bat
   .venv\Scripts\activate.bat
   ```
6. Install dependencies:
   ```powershell
   python -m pip install -r requirements.txt
   ```
7. Run the dashboard:
   ```powershell
   streamlit run app.py
   ```
8. Open the local URL shown in the terminal (usually http://localhost:8501).

## Data files
- `data/customers.csv`: customer IDs, anonymous codes and signup dates.
- `data/orders.csv`: order IDs, customer IDs, dates and statuses.
- `data/order_items.csv`: line item quantities and unit prices.

The app uses the demo dataset by default. To use your own data, upload all three CSV files in the sidebar with the same required column names.

## MySQL setup
Run `sql/schema.sql` in MySQL Workbench or the MySQL command line. The Streamlit app currently reads CSV files; the schema is included as the database design deliverable and can be connected in a later enhancement.

## Methodology
- Recency = days since the most recent completed purchase at the analysis date.
- Frequency = count of distinct completed orders.
- Monetary = sum of quantity × unit price across completed orders.
- Cancelled and returned orders are excluded from spending calculations.
- Rule-based segment labels use transparent thresholds and are editable in `app.py`.
- K-means uses `log1p` for frequency and monetary value, then StandardScaler. K is configurable from 2 to 8. The silhouette score is shown as a rough separation metric, not proof that clusters are meaningful.
- The predictive section creates historical features before a cutoff and predicts spend during the following 90 days. MAE and RMSE are in currency units; R² may be unstable on small or synthetic data.

## Important limitations
- The included dataset is synthetic and for education/demo use only.
- Segments depend on selected dates, thresholds and data quality.
- K-means cluster numbers are arbitrary.
- The demo prediction is not financial advice or a reliable forecast. Real deployment needs larger representative data, temporal validation, monitoring, and privacy controls.
