from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def iris_prediction():
    if request.method == 'GET':
        return render_template("index.html")
    elif request.method == 'POST':
        iris_features = request.form.get("weight")
        radio = request.form["gender"]
        if radio == 'male':
            gender = 'MALE'
        else:
            gender = 'FEMALE'
        print(gender)
        model= joblib.load(f"model/{gender}-classification-using-logistic-regression.pkl")
        print(iris_features)
        result = model.predict([[iris_features]])
        result
        return render_template('index.html', result=result)
    else:
        return "Unsupported Request Method"


if __name__ == '__main__':
    app.run(port=5000, debug=True)
