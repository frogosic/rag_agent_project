import time

from flask import Flask, request, jsonify
from model import AIModel
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
ai = AIModel(policy_path="qa_policy.txt")

SYSTEM_PROMPT = "You are an AI assistant helping with customer inquiries. Provide a helpful and concise response."


@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    user_message = data.get("message")

    if not user_message:
        return jsonify({"error": "Missing message"}), 400

    start_time = time.time()

    try:
        result = ai.get_response(SYSTEM_PROMPT, user_message)
        result["duration"] = round(time.time() - start_time, 3)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
