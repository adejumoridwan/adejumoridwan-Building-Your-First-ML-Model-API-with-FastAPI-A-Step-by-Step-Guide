import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def prepare_dataset():
    """Load and prepare the dataset for modeling"""
    # Load the tips dataset from seaborn
    tips = sns.load_dataset('tips')
    
    print("Dataset overview:")
    print(tips.head())
    print("\nDataset info:")
    print(tips.info())
    
    # Convert categorical features to dummy variables
    tips_encoded = pd.get_dummies(tips, columns=['sex', 'smoker', 'day', 'time'])
    
    # Define features and target
    X = tips_encoded.drop('tip', axis=1)
    y = tips_encoded['tip']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, X.columns

def train_model(X_train, y_train):
    """Train a RandomForestRegressor model"""
    # Create a pipeline with preprocessing and model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    return pipeline


def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model"""
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Evaluation:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    return mse, r2



def save_model(model, filename="tip_predictor_model.joblib"):
    """Save the trained model to a file"""
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")



def prepare_and_train_model():
    X_train, X_test, y_train, y_test, feature_names = prepare_dataset()
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model)
    
    # Save feature names for input validation
    joblib.dump(feature_names, "feature_names.joblib")
    
    return model, feature_names

model, feature_names = prepare_and_train_model()

