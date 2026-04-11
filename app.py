from flask import Flask, request, jsonify, send_from_directory
from src.cascade import PhishingCascade
from src.eml_parser import parse_eml

app = Flask(__name__, static_folder="web")

# Load once at startup — transformer takes a few seconds to load
cascade = PhishingCascade()


@app.route("/")
def index():
    return send_from_directory("web", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid or missing JSON body"}), 400
    result = cascade.predict({
        "from_address":          data.get("from_address", ""),
        "subject":               data.get("subject", ""),
        "body":                  data.get("body", ""),
        "attachment_extensions": data.get("attachment_extensions", []),
        "dkim_pass":             data.get("dkim_pass", False),
        "dkim_domain":           data.get("dkim_domain", None),
    })
    return jsonify(result)


@app.route("/parse_eml", methods=["POST"])
def parse_eml_route():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f = request.files["file"]
    if not f.filename.lower().endswith(".eml"):
        return jsonify({"error": "File must be a .eml file"}), 400
    try:
        result = parse_eml(f.read())
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False)
