import joblib
import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from models import NN_Alzheimer


def preprocess_alzheimer_dataset() -> pd.DataFrame:
    df = pd.read_csv('C:/Users\Denis\Desktop/recognition-app/recognition-app/src/dataset/Alzheimer/dataset.csv')

    # Replacing 0 values
    df.Gender = df.Gender.replace(0,np.nan)
    df.Age = df.Age.replace(0,np.nan)
    df.EDUC = df.EDUC.replace(0,np.nan)
    df.SES = df.SES.replace(0,np.nan)
    df.MMSE = df.MMSE.replace(0,np.nan)
    df.CDR = df.CDR.replace(0,np.nan)
    df.eTIV = df.eTIV.replace(0,np.nan)
    df.nWBV = df.nWBV.replace(0,np.nan)
    df.ASF = df.ASF.replace(0,np.nan)

    # Fill missing values with column means
    df.Gender = df.Gender.fillna(df.Gender.mean())
    df.Age = df.Age.fillna(df.Age.mean())
    df.EDUC = df.EDUC.fillna(df.EDUC.mean())
    df.SES = df.SES.fillna(df.SES.mean())
    df.MMSE = df.MMSE.fillna(df.MMSE.mean())
    df.CDR = df.CDR.fillna(df.CDR.mean())
    df.eTIV = df.eTIV.fillna(df.eTIV.mean())
    df.nWBV = df.nWBV.fillna(df.nWBV.mean())
    df.ASF = df.ASF.fillna(df.ASF.mean())

    df_scaled = preprocessing.scale(df)
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
    df_scaled['Group'] = df['Group']

    return df_scaled


def train_alzheimer():
    df = preprocess_alzheimer_dataset()

    X = df.loc[:, df.columns != 'Group']
    Y = df.loc[:, 'Group']

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)

    model = NN_Alzheimer()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=200)

    scores = model.evaluate(X_train, y_train)
    print("Training Accuracy: %.2f%%\n" % (scores[1] * 100))
    scores = model.evaluate(X_test, y_test)
    print("Testing Accuracy: %.2f%%\n" % (scores[1] * 100))
    model.save('C:/Users/Denis\Desktop/recognition-app/recognition-app/src/webapp/models/alzheimer/model.h5')
    joblib.dump(scaler, 'C:/Users/Denis/Desktop/recognition-app/recognition-app/src/webapp/models/alzheimer/scaler.pkl')


if __name__ == '__main__':
    train_alzheimer()
