import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from preprocess import FeatureEngineering  # Import feature engineering class

def train_model(folder_path):
    # Load and process data
    fe = FeatureEngineering(folder_path)
    X, y = fe.feature_engineering()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Handle class imbalance
    scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
    
    # Define XGBoost model with early stopping
    model = XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=0.1, reg_alpha=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss", use_label_encoder=False,
        random_state=42
    )
    
    # Train with early stopping
    model.fit(X_train, y_train, 
              eval_set=[(X_test, y_test)], 
              early_stopping_rounds=20, 
              verbose=True)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    # Save model
    with open("xgboost_model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    print("Model saved as xgboost_model.pkl")
    
if __name__ == "__main__":
    train_model("data/processed")
