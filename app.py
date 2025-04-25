import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import joblib
from utilis import prepare_and_train_model


app = FastAPI(
    title="Restaurant Tip Predictor API",
    description="API for predicting tips using the Seaborn tips dataset",
    version="1.0.0"
)

class TipPredictionRequest(BaseModel):
    total_bill: float = Field(..., description="Total bill amount", gt=0)
    size: int = Field(..., description="Party size", gt=0)
    sex_Female: bool = Field(False, description="Customer is female")
    sex_Male: bool = Field(False, description="Customer is male")
    smoker_No: bool = Field(False, description="Non-smoker")
    smoker_Yes: bool = Field(False, description="Smoker")
    day_Fri: bool = Field(False, description="Friday")
    day_Sat: bool = Field(False, description="Saturday")
    day_Sun: bool = Field(False, description="Sunday")
    day_Thur: bool = Field(False, description="Thursday")
    time_Dinner: bool = Field(False, description="Dinner time")
    time_Lunch: bool = Field(False, description="Lunch time")
    
    class Config:
        schema_extra = {
            "example": {
                "total_bill": 24.50,
                "size": 4,
                "sex_Female": False,
                "sex_Male": True,
                "smoker_No": True,
                "smoker_Yes": False,
                "day_Fri": False,
                "day_Sat": True,
                "day_Sun": False,
                "day_Thur": False,
                "time_Dinner": True,
                "time_Lunch": False
            }
        }

class TipPredictionResponse(BaseModel):
    predicted_tip: float

def load_model(model_path="tip_predictor_model.joblib"):
    """Load the trained model from file"""
    try:
        model = joblib.load(model_path)
        return model
    except:
        raise HTTPException(
            status_code=500, 
            detail="Model not found. Please train the model first."
        )


model = None
feature_names = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global model, feature_names
    try:
        model = joblib.load("tip_predictor_model.joblib")
        feature_names = joblib.load("feature_names.joblib")
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Training new model...")
        model, feature_names = prepare_and_train_model()

@app.get("/")
def read_root():
    """Root endpoint"""
    return {"message": "Welcome to the Restaurant Tip Predictor API"}

@app.post("/predict", response_model=TipPredictionResponse)
def predict_tip(request: TipPredictionRequest):
    """Predict tip amount based on input features"""
    # Convert input data to DataFrame with correct columns
    input_dict = request.model_dump()
    input_df = pd.DataFrame([input_dict])
    
    # Reorder columns to match the model's expected feature names
    try:
        input_df = input_df[feature_names]
    except KeyError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Input data is missing required features: {e}"
        )
    
    # Make prediction
    predicted_tip = model.predict(input_df)[0]
    
    return TipPredictionResponse(
        predicted_tip=round(float(predicted_tip), 2)
    )

if __name__ == "__main__":
    import uvicorn
    try:
        model = joblib.load("tip_predictor_model.joblib")
        feature_names = joblib.load("feature_names.joblib")
    except:
        print("No model found. Training new model...")
        model, feature_names = prepare_and_train_model()
        
    # Run the API server
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

# To run this API, save this file as tip_predictor.py and run:
# python tip_predictor.py

# Once running, access the API documentation at:
# http://localhost:8000/docs