from flask import Flask, render_template, request

from utils import predict_diabetes, predict_alzheimer


app = Flask(__name__)


@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    result = None

    if request.method == 'POST':
        glucose = float(request.form['glucose'])
        blood_pressure = float(request.form['blood_pressure'])
        skin_thickness = float(request.form['skin_thickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        pregnancies = float(request.form['pregnancies'])
        diabetes_pedigree = float(request.form['diabetes_pedigree'])
        age = float(request.form['age'])

        prediction = predict_diabetes(
            glucose, blood_pressure, skin_thickness, insulin,
            bmi, pregnancies, diabetes_pedigree, age
        )

        result = 'Positive' if prediction > 0.5 else 'Negative'

    return render_template('diabetes.html', result=result)


@app.route('/alzheimer', methods=['GET', 'POST'])
def alzheimer():
    result = None

    if request.method == 'POST':
        Gender = float(request.form['Gender'])
        Age = float(request.form['Age'])
        Educ = float(request.form['EDUC'])
        SES = float(request.form['SES'])
        MMSE = float(request.form['MMSE'])
        CDR = float(request.form['CDR'])
        eTIV = float(request.form['eTIV'])
        nWBV = float(request.form['nWBV'])
        ASF = float(request.form['ASF'])

        prediction = predict_alzheimer(
            Gender,Age,Educ,SES,MMSE,CDR,eTIV,nWBV,ASF
        )

        result = 'Positive' if prediction > 0.5 else 'Negative'

    return render_template('alzheimer.html', result=result)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
