from flask import Flask, request, jsonify
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense
import joblib
from business_logic import urgency_score, root_cause

app = Flask(__name__)

MAX_LEN = 100
VOCAB_SIZE = 10000
EMBED_DIM = 100
NUM_CLASSES = 4

tokenizer = joblib.load("tokenizer.joblib")

model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_DIM),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.3),
    Dense(NUM_CLASSES, activation="softmax")
])

model.build(input_shape=(None, MAX_LEN))
model.load_weights("customer_model-003.keras")

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "API running"})

def preprocess_text(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=MAX_LEN, padding="post")
    return padded

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = data["text"]
    seq = preprocess_text(text)

    pred_probs = model.predict(seq)
    pred_class = int(tf.argmax(pred_probs, axis=1)[0])

    categories = ["billing", "technical", "service", "fraud", "general"]
    category = categories[pred_class]

    return jsonify({
        "complaint": text,
        "category": category,
        "urgency": urgency_score(text, category),
        "root_cause": root_cause(text)
    })

@app.route("/predict_csv", methods=["POST"])
def predict_csv():
    if "file" not in request.files:
        return jsonify({"error": "CSV file missing"}), 400

    file = request.files["file"]
    text_column = request.form.get("text_column", "complaint")

    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"Failed to read CSV: {str(e)}"}), 400

    if text_column not in df.columns:
        return jsonify({"error": "Invalid text column"}), 400

    categories = ["billing", "technical", "service", "fraud", "general"]

    def predict_row(text):
        seq = preprocess_text(str(text))
        pred_probs = model.predict(seq)
        pred_class = int(tf.argmax(pred_probs, axis=1)[0])
        return categories[pred_class]

    df["predicted_category"] = df[text_column].apply(predict_row)
    df["urgency"] = df.apply(
        lambda x: urgency_score(x[text_column], x["predicted_category"]), axis=1
    )
    df["root_cause"] = df[text_column].apply(root_cause)

    return jsonify(df.to_dict(orient="records"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
