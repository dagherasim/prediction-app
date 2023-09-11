import joblib
import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from models import NN


def preprocess_diabetes_dataset() -> pd.DataFrame:
    df = pd.read_csv('../dataset/Diabetes/dataset.csv')

    # Replacing 0 values
    df.Glucose = df.Glucose.replace(0, np.nan)
    df.BloodPressure = df.BloodPressure.replace(0, np.nan)
    df.SkinThickness = df.SkinThickness.replace(0, np.nan)
    df.Insulin = df.Insulin.replace(0, np.nan)
    df.BMI = df.BMI.replace(0, np.nan)
    df.Pregnancies = df.Pregnancies.replace(0, np.nan)
    df.DiabetesPedigreeFunction = df.DiabetesPedigreeFunction.replace(0, np.nan)
    df.Age = df.Age.replace(0, np.nan)

    # Fill missing values with column means
    df.Glucose = df.Glucose.fillna(df.Glucose.mean())
    df.BloodPressure = df.BloodPressure.fillna(df.BloodPressure.mean())
    df.SkinThickness = df.SkinThickness.fillna(df.SkinThickness.mean())
    df.Insulin = df.Insulin.fillna(df.Insulin.mean())
    df.BMI = df.BMI.fillna(df.BMI.mean())
    df.Pregnancies = df.Pregnancies.fillna(df.Pregnancies.mean())
    df.DiabetesPedigreeFunction = df.DiabetesPedigreeFunction.fillna(df.DiabetesPedigreeFunction.mean())
    df.Age = df.Age.fillna(df.Age.mean())

    df_scaled = preprocessing.scale(df)
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
    df_scaled['Outcome'] = df['Outcome']

    return df_scaled


def train_diabetes():
    df = preprocess_diabetes_dataset()

    X = df.loc[:, df.columns != 'Outcome']
    Y = df.loc[:, 'Outcome']

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)

    model = NN()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=200)

    scores = model.evaluate(X_train, y_train)
    print("Training Accuracy: %.2f%%\n" % (scores[1] * 100))
    scores = model.evaluate(X_test, y_test)
    print("Testing Accuracy: %.2f%%\n" % (scores[1] * 100))
    model.save('./saved_models/diabetes/model.h5')
    joblib.dump(scaler, './saved_models/diabetes/scaler.pkl')


if __name__ == '__main__':
    train_diabetes()
