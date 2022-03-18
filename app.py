from flask import Flask, render_template, request
import numpy as np
from joblib import dump,load

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
	if request.method == "POST":
		news = request.form["news"]
	if news == "":
		res = {'null':'null'}
	else:
		vect = load('vectorizer.pkl')
		scaler = load('scaler.pkl')
		mnb = load('model.pkl')
		le = load('le.pkl')
		text_v = vect.transform([news])
		text_std = scaler.transform(text_v)
		prediction = mnb.predict(text_std)
		val = le.inverse_transform(prediction)[0]
		res = {val:'null'}
	return render_template('index.html',pred=res.items())
		 

if __name__ == "__main__":
	app.run() 