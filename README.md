# 🩺 Disease Prediction System (ML + API)

A machine learning–powered disease prediction system exposed via a REST API.  
This project demonstrates **end-to-end ML workflow** — from data preprocessing and model training to deployment-ready inference using **FastAPI**.

Built with simplicity and clarity in mind, especially for real-world usage and interviews.

---
## Video Demo
https://youtu.be/ejK7z4ae-2s
## 🚀 Features

- ✅ Predicts disease likelihood based on medical input parameters
- ✅ RESTful API using **FastAPI**
- ✅ Supports both:
  - `.predict()` → final prediction  
  - `.predict_proba()` → probability / confidence score
- ✅ Clean input validation using **Pydantic**
- ✅ Production-ready project structure
- ✅ Easily extendable to mobile apps (Android / Web)

---

## 🧠 Tech Stack

### 🎨 Frontend (Client Application) 
The frontend acts as a **client layer** that collects user medical inputs and communicates with the backend ML API to display predictions and confidence scores. 

The system is designed to be **backend-agnostic**, meaning any frontend (mobile, web, desktop) can consume the API.

### 📱 Android Frontend (Jetpack Compose) 
#### Tech Stack 
- **Kotlin** 
- **Jetpack Compose** 
- **ViewModel** (state management) 
- **Repository** 
- **Coroutines** (asynchronous calls)

## 📱 Android App Structure

```text
com.example.diseaseprediction
│
├── Repository
│   └── NetworkRepository.kt
│
├── Retrofit
│   ├── Request_and_Responses
│   │   ├── BrainTumour.kt
│   │   ├── Breast.kt
│   │   ├── Diebetes.kt
│   │   └── Heart.kt
│   │
│   ├── Endpoints.kt
│   └── RetrofitInstance.kt
│
├── ui.theme
│
├── View
│   ├── BrainTumorPrediction.kt
│   ├── BreastPrediction.kt
│   ├── DiabetesPrediction.kt
│   ├── HeartDiseasePrediction.kt
│   └── HomePage.kt
│
├── ViewModel
│   └── MyViewModel.kt
│
└── MainActivity.kt
```

### Machine Learning
- **Python** 
- **Tensorflow** 
- **scikit-learn** 
- **NumPy** 
- **Pandas** 
- **Joblib / Pickle** (model serialization)

### Backend / API
- **FastAPI**
- **Uvicorn**
- **Pydantic**

### Deployment Ready
- Works on cloud platforms (Render / local server)

---

## 📁 Project Structure

```text
├── model/
│   ├── trained_model.pkl
│   └── scaler.pkl
│
├── app/
│   ├── main.py          # FastAPI entry point
│   ├── schemas.py       # Pydantic request/response models
│   └── utils.py         # Model loading & prediction logic
│
├── requirements.txt
├── README.md
└── train_model.py       # Model training pipeline
```
### 🎗 Breast Cancer Prediction

#### Endpoint
### REQUEST
```http
POST /BreastPrediction
{
  "types": [
    17.99, 10.38, 122.8, 1001.0, 0.1184,
    0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
    1.095, 0.9053, 8.589, 153.4, 0.006399,
    0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
    25.38, 17.33, 184.6, 2019.0, 0.1622,
    0.6656, 0.7119, 0.2654, 0.4601, 0.1189
  ]
}
```
### RESPONSE
```
{
  "result": "Malignant",
  "prediction": "You have cancer",
  "chances": 78
}
```
### 🩸 Diabetes Prediction

#### Endpoint
### REQUEST

```http
POST /DiebetesPrediction
{
  "age": 45,
  "bmi": 32.5,
  "hbAc": 6.1,
  "glucose": 148,
  "smoke": 1,
  "hypertension": 0,
  "heart": 0
}
```
### RESPONSE

```{
  "prediction": 1,
  "probability": 0.82
}
```

---

### ❤️ Heart Disease Prediction

#### Endpoint 
### REQUEST

```http
POST /HeartPrediction
{
  "heart_input": [
    52, 1, 0, 125, 212,
    0, 1, 168, 0, 1.0,
    2, 0, 2
  ]
}
```
### RESPONSE

```
{
  "status": "You have heart problem",
  "chances": 67
}
```
---

### 🧠 Brain Tumor Prediction

#### Endpoint
```http
POST /BrainPrediction
{
  "result": "You do not have tumour",
  "chances": 55
}
```
