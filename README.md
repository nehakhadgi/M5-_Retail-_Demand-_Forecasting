# M5 Retail Demand Forecasting Dashboard

An interactive, enterprise-grade demand forecasting application utilizing baseline and challenger machine learning models to optimize retail supply chain decisions.

## 📊 Project Overview
This project tackles the **M5 Forecasting Challenge** dataset to predict 28 days of daily unit sales. Our system bridges the gap between statistical accuracy and operational decisions by calculating asymmetric inventory costs.

## 🚀 Features
- **Data Pipeline**: Streamlined memory-conscious loading (using custom chunk aggregation) of evaluation records.
- **Forecasting Models**:
  - **Seasonal Naive (Baseline)**: A designed hurdle replicating the historical week-over-week seasonal patterns.
  - **SARIMA**: Seasonal Autoregressive Integrated Moving Average model capturing linear trends and weekly seasonality.
  - **Prophet**: Additive regression model handling strong seasonal patterns and holiday effects.
- **Asymmetric Cost Evaluation**: Adjusts forecast assessments by pricing inventory shortage ($8/unit underage cost) versus overstocking ($2/unit overage cost).
- **Interactive UI**: Slider inputs dynamically shift operational parameters to recalculate model scorecards in real time.

## 🛠️ Tech Stack & Requirements
- **Python 3.12**
- **Libraries**:
  - `numpy==2.5.4`
  - `pandas==3.0.6`
  - `matplotlib==3.11.2`
  - `statsmodels==0.15.0`
  - `prophet==1.5.0`
  - `streamlit`

## ⚙️ Running Locally
1. Clone this repository:
   ```bash
   git clone https://github.com/nehakhadgi/M5-_Retail-_Demand-_Forecasting.git
   cd M5-_Retail-_Demand-_Forecasting
   ```
2. Set up a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the dashboard:
   ```bash
   streamlit run app.py
   ```

## 📈 Production Plan
- **Retraining Cadence**: Models refresh every 28 days as new actuals arrive.
- **Monitoring**: Continuous tracking of rolling 28-day MAE and signed bias.
- **Alerting**: System triggers alerts when rolling MAE exceeds the baseline/backtest champion's MAE by more than 20%.
- **Fallback**: Automated fallback to **Seasonal Naive** if primary compute nodes or upstream dependencies become unavailable.
