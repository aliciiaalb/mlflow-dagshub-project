from flask import Flask, request, jsonify
import mlflow.pyfunc

app = Flask(__name__)

# Charger le modèle MLflow
model = mlflow.pyfunc.load_model("model")  # Assure-toi que le dossier "model" est bien présent

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    predictions = model.predict(data["inputs"])
    return jsonify({"predictions": predictions.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
