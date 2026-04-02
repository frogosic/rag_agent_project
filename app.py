import logging
import os
import time

from flask import Flask, request, jsonify, render_template, session
from model import AIModel
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")

ai = AIModel(
    policy_path="qa_policy.txt",
    collection_name="qa_policies",
    doc_metadata={"domain": "qa", "version": "2024-Q1"},
)

SYSTEM_PROMPT = (
    "You are a QA assistant helping engineers understand quality assurance policies. "
    "Answer questions based on the provided policy context. "
    "If the answer is not in the context, say you don't know."
)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    user_message = data.get("message")

    if not user_message:
        return jsonify({"error": "Missing message"}), 400

    if "history" not in session:
        session["history"] = []

    start_time = time.time()

    try:
        result = ai.get_response(SYSTEM_PROMPT, user_message, session["history"])

        session["history"].append(
            {"user": user_message, "assistant": result.get("response", "")}
        )
        session.modified = True

        result["duration"] = round(time.time() - start_time, 3)
        return jsonify(result)
    except Exception as e:
        logger.error("Error generating response: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/reset", methods=["POST"])
def reset():
    session.pop("history", None)
    return jsonify({"message": "Conversation history cleared"}), 200


if __name__ == "__main__":
    app.run(debug=True)
