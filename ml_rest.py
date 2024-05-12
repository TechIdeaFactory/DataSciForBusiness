# pip install fastapi ‘uvicorn[standard]’
# pip install pandas

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the saved RandomForest model
rf_model = joblib.load('random_forest_model.pkl')

# Define input data model
class InputData(BaseModel):
   bmi: int
   hba1c_level: float
   smoke_current: int
   smoke_ever: int


# Create FastAPI app instance
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Diabetes Classifier ML API"}


@app.post('/diabetes/predict')
def predict_diabetes(data: InputData):
    data = data.dict()
    bmi = data['bmi']
    hba1c_level = data['hba1c_level']
    smoke_current = data['smoke_current']
    smoke_ever = data['smoke_ever']

    d = {'bmi': [bmi],
         'HbA1c_level': [hba1c_level],
         'smoking_history_current': [smoke_current],
         'smoking_history_never': [smoke_ever]}

    df = pd.DataFrame(data=d)
    prediction = rf_model.predict(df)

    return {
        'prediction': prediction.item(0)
    }

if __name__ == "__main__":
   import uvicorn

uvicorn.run(app, host="localhost", port=8000)
