import joblib
import pandas as pd

from keras.models import load_model


DIABETES_MODEL = load_model('src/webapp/models/diabetes/model.h5')
DIABETES_SCALER = joblib.load('src/webapp/models/diabetes/scaler.pkl')
ALZHEIMER_MODEL =  load_model('src/webapp/models/alzheimer/model.h5')
ALZHEIMER_SCALER = joblib.load('src/webapp/models/alzheimer/scaler.pkl')


def predict_diabetes(
        glucose: float = 0.0,
        blood_pressure: float = 0.0,
        skin_thickness: float = 0.0,
        insulin: float = 0.0,
        bmi: float = 0.0,
        pregnancies: float = 0.0,
        diabetes_pedigree: float = 0.0,
        age: float = 0.0,
) -> int:
    user_data = pd.DataFrame({
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'Pregnancies': [pregnancies],
        'DiabetesPedigreeFunction': [diabetes_pedigree],
        'Age': [age]
    })

    user_data_scaled = DIABETES_SCALER.transform(user_data)
    prediction = DIABETES_MODEL.predict(user_data_scaled)[0][0]

    return prediction


def predict_alzheimer(
        Gender: float = 0.0,
        Age: float = 0.0,
        EDUC: float = 0.0,
        SES: float = 0.0,
        MMSE: float = 0.0,
        CDR: float = 0.0,
        eTIV: float = 0.0,
        nWBV: float = 0.0,
        ASF: float = 0.0,
) -> int:
    user_data = pd.DataFrame({
        'Gender': [Gender],
        'Age': [Age],
        'EDUC': [EDUC],
        'SES': [SES],
        'MMSE': [MMSE],
        'CDR': [CDR],
        'eTIV': [eTIV],
        'nWBV': [nWBV],
        'ASF':[ASF]
    })

    user_data_scaled = ALZHEIMER_SCALER.transform(user_data)
    prediction = ALZHEIMER_MODEL.predict(user_data_scaled)[0][0]

    return prediction