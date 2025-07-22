from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('predictive.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def recomend_crop():
    age = int(request.form.get('age'))
    anaemia = int(request.form.get('anaemia'))
    creatinine_phosphokinase = float(request.form.get('creatinine_phosphokinase'))
    diabetes = int(request.form.get('diabetes'))
    ejection_fraction = float(request.form.get('ejection_fraction'))
    high_blood_pressure = int(request.form.get('high_blood_pressure'))
    platelets = float(request.form.get('platelets'))
    serum_creatinine = float(request.form.get('serum_creatinine'))
    serum_sodium = float(request.form.get('serum_sodium'))
    sex = int(request.form.get('sex'))
    smoking = int(request.form.get('smoking'))
    time = str(request.form.get('time'))
    

    # prediction
    death_event = [0,1]
    result = model.predict(np.array([age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, platelets,
       serum_creatinine, serum_sodium, sex, smoking, time]).reshape(1,12))
    index = result[0] - 1
    # image = Image.open(f"crops/{crops[index]}.jpg")
    # image.show()
    result = str(death_event[index])
    return render_template('index.html', result = result)


if __name__ == '__main__':
    app.run(debug = True)