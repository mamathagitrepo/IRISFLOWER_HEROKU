import joblib
model = joblib.load('model_saved.pkl')
prediction = model.predict([[4.5,3.0,4.5,2.8]])

print(prediction)