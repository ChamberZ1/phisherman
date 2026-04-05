from flask import Flask, request, jsonify, send_from_directory
from src.cascade import PhishingCascade

app = Flask(__name__, static_folder="web")

# Load once at startup — transformer takes a few seconds to load
cascade = PhishingCascade()


@app.route("/")
def index():
    return send_from_directory("web", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    result = cascade.predict({
        "from_address": data.get("from_address", ""),
        "subject": data.get("subject", ""),
        "body": data.get("body", ""),
    })
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=False)
