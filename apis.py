import mlflow
import joblib
import pandas as pd
from fastapi import Body
from fastapi import FastAPI
from fastapi.responses import JSONResponse



# Create the FastAPI app instance
app = FastAPI()



@app.get("/")
def home():
    return {"message": "Welcome to Home"}



@app.get("/health")
def check_health():
    return {"status": "healthy"}



@app.post("/predict")
def predict(data: dict = Body(...)):
    


    model = joblib.load("model.pkl")

    # Load the pipeline 
    transformer_pipeline = joblib.load("transformer.pkl")

    # Drop unnecessary columns for prediction
    features = pd.DataFrame(data).drop(columns=['CustomerId', 'Surname'])

    # Transform the data
    transformed = transformer_pipeline.transform(features)

    # Predict on a Pandas DataFrame (make inference)
    pred = model.predict(pd.DataFrame(transformed))


    return JSONResponse(content={"prediction": f"{pred.tolist()}"})
