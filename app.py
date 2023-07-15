from flask import Flask,render_template,request
import joblib
# initialize the app

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict',methods=['post'])
def predict():

    model = joblib.load('model_saved.pkl')
    sepallength = request.form.get("sepallength")
    sepalwidth = request.form.get("sepalwidth")
    petallength = request.form.get("petallength")
    petalwidth = request.form.get("petalwidth")

    print(sepallength,sepalwidth,petallength,petalwidth)
    output = model.predict([[float(sepallength),float(sepalwidth),float(petallength),float(petalwidth)]])
    data = output

    return render_template('predict.html',data=data)

if __name__ == '__main__':

    app.run(debug=True)

