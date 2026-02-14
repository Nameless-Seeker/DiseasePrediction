# import io
import pickle
from typing import List
import pandas as pd
from PIL import Image
from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel, Field
import tensorflow as tf

disease = FastAPI()

# Load dictionary
model_data = joblib.load("diabetes_model.joblib")

# Extract components
model = model_data["model"]
imputer = model_data["imputer"]
scaler = model_data["scaler"]
encoder = model_data["encoder"]

numeric_cols = model_data["numeric_cols"]
categorical_cols = model_data["categorical_cols"]

#Diebetes Data validation
class DiabetesInput(BaseModel):
    age: float
    bmi: float
    hbAc: float              # HbA1c_level
    glucose: float         # blood_glucose_level
    smoke: int             # 1 = yes, 0 = no
    hypertension: int             # hypertension
    heart: int


def smokeCategory(data: DiabetesInput):
    smokeInfo = data.smoke

    if (smokeInfo == 1):
        return "former"

    elif (smokeInfo == 0):
        return "never"

    elif (smokeInfo == 2):
        return "current"

    elif (smokeInfo == 3):
        return "not current"

    elif (smokeInfo == 4):
        return "ever"

    else:
        return "No Info"

@disease.get("/")
def health():
    return {'Status':"OK"}

#Diebetes prediction
@disease.post("/DiebetesPrediction")
def predict_diabetes(data: DiabetesInput):

    # Build input dataframe
    df = pd.DataFrame([{
        "age": data.age,
        "hypertension": data.hypertension,
        "heart_disease": data.heart,
        "bmi": data.bmi,
        "HbA1c_level": data.hbAc,
        "blood_glucose_level": data.glucose,
        "smoking_history": smokeCategory(data)
    }])

    # 1. Numeric → impute
    df_num = imputer.transform(df[numeric_cols])

    # 2. Numeric → scale
    df_num_scaled = scaler.transform(df_num)

    # 3. Categorical → encode
    df_cat = encoder.transform(df[categorical_cols])

    # 4. Combine numeric + encoded categorical
    final_features = np.concatenate([df_num_scaled, df_cat], axis=1)

    # 5. Predict class
    prediction = model.predict(final_features)[0]

    # 6. Predict probability (class = 1, diabetes)
    probability = model.predict_proba(final_features)[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }

######################################################################
# Breast Cancer Prediction
class BreastPrediction(BaseModel):
    types: List[float] = Field(..., min_length=30, max_length=30)

# Loading data
with open("BreastCancer.pkl", "rb") as cancer:
    breast_model = pickle.load(cancer)

with open("Scaler.pkl", 'rb') as f:
    breast_scaler = pickle.load(f)


# Predicting cancer
def predict_breast_cancer(model, scaler, input_array: list[float]):
    # Converting the list to numpy array
    numpy_array = np.array(input_array).reshape(1, -1)
    # Scalling the numpy array by scaler
    scaled_input_array = scaler.transform(numpy_array)

    # Getting the begun,magigant chances 2D array [[0.8875,0.4567]]
    # Prediction
    ans = model.predict_proba(scaled_input_array)[0]

    return ans


#Breast cancer prediction
@disease.post("/BreastPrediction")
def Breast_Prediction(data: BreastPrediction):
    # Getting the Prediction: [[0.8875,0.4567]]
    probability_list = predict_breast_cancer(breast_model, breast_scaler, data.types)

    begun = int(probability_list[0]*100)    #0.8875*100 = 88.75 = 88
    malig = int(probability_list[1]*100)    #0.4567*100 = 45.67 = 45

    result = {}

    if (begun > malig):
        result["result"] = "Begun"
        result["prediction"] = "You do not have cancer"
        result["chances"] = begun

    else:
        result["result"] = "Malignant"
        result["prediction"] = "You have cancer"
        result["chances"] = malig

    return result
##################################################################################################
#Heart disease prediction
heart_model_load = joblib.load("heart_model.pkl")
    
# print(type(heart_model))

heart_model = heart_model_load["model"]
heart_model_scaler = heart_model_load["scaler"]

# srijan = {
#     "heart_input": [
#         52, 1, 0, 125, 212,
#         0, 1, 168, 0, 1.0,
#         2, 0, 2
#     ]
# }

class HeartPrediction(BaseModel):
    heart_input: List[float] = Field(..., max_length=13, min_length=13)

def predict_heart_diesease(model,scaler, heart_data):    
    heart_transform = np.array(heart_data).reshape(1, -1)
    heart_transform_scaled = scaler.transform(heart_transform)

    
    prediction = model.predict_proba(heart_transform_scaled)[0]

    return prediction


@disease.post("/HeartPrediction")
def heart_prediction(heart_input_data: HeartPrediction):
    prediction_result = predict_heart_diesease(heart_model,heart_model_scaler, heart_input_data.heart_input)

    heart_problem = int(prediction_result[1]*100)
    no_heart_problem = int(prediction_result[0]*100)

    result = {}

    if (heart_problem > no_heart_problem):
        result["status"] = "You have heart problem"
        result["chances"] = heart_problem
    else:
        result["status"] = "You do not have heart problem"
        result["chances"] = no_heart_problem
        
    return result
#######################################################################################
# Brain tumour detection

load_model = tf.keras.models.load_model("brain.keras")
# print("Model loaded successfully")
# # print(model.summary())

def PredictBrainTumour(image):
    # Preprocess
    image = image.resize((64,64))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    prediction = load_model.predict(image_array)[0]
    return prediction


@disease.post("/BrainPrediction")
async def Brain(file: UploadFile = File(...)):
    contents = await file.read()

    # Convert image into model readable image
    image = Image.open(io.BytesIO(contents)).convert("RGB")


    # Getting the prediction confidence [0.4512,0.6554]
    prediction = PredictBrainTumour(image)

    # Converting the confidence to integer % -> [45,55]
    tumour = int(prediction[1]*100)     # -> 45
    no_tumour = int(prediction[0]*100)  # -> 55

    result = {}

    # Comparing the results
    if(tumour > no_tumour):
        result['result'] = "You have tumour"
        result['chances'] = tumour
    else:
        result['result'] = "You do not have tumour"
        result['chances'] = no_tumour

    print(result)

    return result