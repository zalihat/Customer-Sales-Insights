import pandas as pd
import os
from datetime import timedelta
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

class FeatureEngineering:
    def __init__(self, silver_path, gold_path):
        """
        Initialize the FeatureEngineering class by loading parquet files.
        """
        self.folder_path = silver_path
        self.gold_path = gold_path
        self.dfs = self.load_dfs_from_parquet()
        self.df = self.merge_tables()
        os.makedirs(self.folder_path, exist_ok=True)
        os.makedirs(self.gold_path, exist_ok=True)
    
    def load_dfs_from_parquet(self):
        """
        Loads all parquet files within a folder into a dictionary of DataFrames.
        """
        dfs = {}
        for filename in os.listdir(self.folder_path):
            if filename.endswith(".parquet"):
                file_path = os.path.join(self.folder_path, filename)
                df_name = filename[:-8]  # Remove ".parquet" from filename
                dfs[df_name] = pd.read_parquet(file_path)
        return dfs
    
    def merge_tables(self):
        """
        Merge all necessary tables into a single DataFrame.
        """
        required_keys = [
        'cleaned_orders_table',
        'cleaned_customer_table',
        'cleaned_order_items_table',
        'cleaned_order_payments_table',
        'cleaned_reviews_table'
        ]

        missing_keys = [key for key in required_keys if key not in self.dfs]
        if missing_keys:
          raise ValueError(f"Missing cleaned tables: {missing_keys}")
        orders = self.dfs['cleaned_orders_table'].dropna()
        customers = self.dfs['cleaned_customer_table'].dropna()
        order_items = self.dfs['cleaned_order_items_table'].dropna()
        payments = self.dfs['cleaned_order_payments_table'].dropna()
        reviews = self.dfs['cleaned_reviews_table'].dropna()
        
        df = orders.merge(customers, on="customer_id", how="left")
        df = df.merge(order_items, on="order_id", how="left")
        df = df.merge(payments, on="order_id", how="left")
        df = df.merge(reviews, on="order_id", how="left")
        return df
    
    def remove_outliers(self, df, column):
        """
        Removes outliers using IQR method.
        """
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    def feature_engineering(self):
        """
        Perform feature engineering on the merged dataset.
        """
        df = self.df.copy()

        # Convert order timestamp to datetime
        df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
        
        # Find last order date per unique customer
        customer_last_order = df.groupby("customer_unique_id")["order_purchase_timestamp"].max().reset_index()
        
        # Define churn threshold (inactive for 6 months)
        churn_threshold = df["order_purchase_timestamp"].max() - timedelta(days=180)
        
        # Label churn (1 if last purchase is older than 6 months, else 0)
        customer_last_order["is_churned"] = (customer_last_order["order_purchase_timestamp"] < churn_threshold).astype(int)
        
        # Merge back
        df = df.merge(customer_last_order[["customer_unique_id", "is_churned"]], on="customer_unique_id", how="left")
        
        # Create customer-level aggregated features
        customer_features = df.groupby("customer_unique_id").agg(
            customer_lifetime_orders=("order_id", "count"),
            customer_avg_order_value=("payment_value", "mean"),
            customer_days_since_last_order=("order_purchase_timestamp", lambda x: (df["order_purchase_timestamp"].max() - x.max()).days)
        ).reset_index()
        
        # Identify high spenders
        spend_threshold = df["payment_value"].quantile(0.90)
        df["is_high_spender"] = (df["payment_value"] > spend_threshold).astype(int)
        
        # Payment features
        customer_payment = df.groupby("customer_unique_id").agg(
            customer_avg_installments=("payment_installments", "mean")
        ).reset_index()
        
        # Customer experience features
        customer_experience = df.groupby("customer_unique_id").agg(
            avg_review_score=("review_score", "mean"),
            total_reviews=("review_id", "count")
        ).reset_index()
        
        # Merge all features into a final dataset
        df = df.merge(customer_features, on="customer_unique_id", how="left")
        df = df.merge(customer_payment, on="customer_unique_id", how="left")
        df = df.merge(customer_experience, on="customer_unique_id", how="left")
        output_file_path = os.path.join(self.gold_path, "model_data.parquet")
        df.to_parquet(output_file_path)

        # Remove outliers
        df = self.remove_outliers(df, "customer_avg_order_value")
        
        # Fill missing values
        imputer = SimpleImputer(strategy="median")
        df[["customer_avg_order_value", "customer_avg_installments", "avg_review_score"]] = imputer.fit_transform(
            df[["customer_avg_order_value", "customer_avg_installments", "avg_review_score"]])

        # Feature Scaling
        scaler = StandardScaler()
        df[["customer_lifetime_orders", "customer_avg_order_value", "customer_avg_installments", "avg_review_score"]] = scaler.fit_transform(
            df[["customer_lifetime_orders", "customer_avg_order_value", "customer_avg_installments", "avg_review_score"]])

        # Select features and target
        features = ["customer_lifetime_orders", "customer_avg_order_value",
                    "customer_avg_installments", "avg_review_score"]
        target = "is_churned"
        # Separate churned and non-churned customers
        df_churned = df[df["is_churned"] == 1]
        df_non_churned = df[df["is_churned"] == 0]
        # Upsample the minority class (non-churned customers)
        df_non_churned_upsampled = resample(df_non_churned, 
                                            replace=True,  # Allow duplicates
                                            n_samples=len(df_churned), 
                                            random_state=42)

        # Combine with churned customers
        df_balanced = pd.concat([df_churned, df_non_churned_upsampled])

        X = df_balanced[features]
        y = df_balanced[target]

        return X, y
