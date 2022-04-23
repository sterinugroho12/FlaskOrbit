from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def iris_prediction():
    if request.method == 'GET':
        return render_template("beratbadan_prediction.html")#gak perlu diganti
    elif request.method == 'POST':
        print(dict(request.form))
        berat = dict(request.form).values()
        berat = np.array([float(x) for x in berat])
        model, std_scaler = joblib.load("model-development/beratbadan-linearregresion.pkl") #ini nggak perlu diganti
        berat = std_scaler.transform([berat])#gak perlu diganti
        print(berat)
        result = model.predict(berat)/10
        return render_template('beratbadan_prediction.html', result=result)
    else:
        return "Unsupported Request Method"


if __name__ == '__main__':
    app.run(port=5000, debug=True)