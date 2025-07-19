# 🧠 Customer Churn Prediction

This project aims to predict customer churn using historical sales data. It follows a modular data pipeline design based on the **Medallion Architecture** (Bronze → Silver → Gold), with clear separation of concerns across ingestion, cleaning, feature engineering, and modeling.

---

## 📁 Project Structure

- **data/**
  - **bronze/** – Raw data ingested from the API
  - **silver/** – Cleaned and transformed data
  - **gold/** - Training data for the Model

- **src/**
  - **data_ingestion.py** – Fetches sales data from an API and stores it in the `data/bronze/` directory
  - **data_cleaning.py** – Cleans the raw data and stores it in the `data/silver/` directory
  - **feature_engineering.py** – Creates features required for model training

- **Notebook/**
  - **EDA.ipynb** – Exploratory Data Analysis notebook

- **doc/**
  - **business Insights and recommendations.py** – Business analysis and actionable recommendations
  - **Customer Churn Prediction - model Evaluation.pdf** – Model performance and evaluation report
  - **Data quality report.pdf** – Data quality assessment

- **requirements.txt** – List of Python packages used in the project
- **README.md** – Project overview and documentation

---

## 🧱 Medallion Architecture

This project follows a **Medallion architecture** to structure the data pipeline:

- **Bronze Layer**: Raw data ingested directly from the API and stored in `/data/bronze`.
- **Silver Layer**: Cleaned, validated data stored in `/data/silver`.
- **Gold Layer**: Feature-rich dataset ready for modeling and analytics stored in `/data/gold`. 

---

## 🚀 Pipeline Overview

1. **Data Ingestion**
   - Source: External API  
   - Output: Raw data saved in `data/bronze/`  
   - Script: `src/data_ingestion.py`

2. **Data Cleaning**
   - Handles missing values, type conversions, and data integrity checks.  
   - Output: Cleaned data saved in `data/silver/`  
   - Script: `src/data_cleaning.py`

3. **Feature Engineering**
   - Extracts and creates features for model training.  
   - Output: Final dataset for modeling  
   - Script: `src/feature_engineering.py`

4. **Exploratory Data Analysis**
   - Conducted in `Notebook/EDA.ipynb` to derive insights and guide modeling.

5. **Model Evaluation**
   - Performance metrics and insights documented in `doc/Customer Churn Prediction - model Evaluation.pdf`.

---

## 📈 Business Insights

Insights and actionable recommendations based on customer churn patterns are documented in:

- `doc/business Insights and recommendations.py`

---

## 📌 Next Steps

- [ ] Orchestrate the pipeline (Airflow / Prefect / Dagster)
- [ ] Implement **incremental loading** strategy for new data
- [ ] Add **partitioning logic** for large-scale data handling
- [ ] Set up model deployment 

---

## 📦 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
🧑‍💻 Contributions
Open to contributions. Feel free to fork and submit a PR.
