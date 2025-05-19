from fastapi import FastAPI
from fastapi.responses import JSONResponse



# Create the FastAPI app instance
app = FastAPI()



# Define the route using app.get (or app.post etc.)
@app.get("/")
def home():
    return {"message": "Welcome to the API"}




@app.get("/health")
def check_health():
    return {"status": "healthy"}




@app.post("/predict")
def predict():
    # 
    return JSONResponse(content={"prediction": "This is a placeholder"})
