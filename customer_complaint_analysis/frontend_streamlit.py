import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
import joblib
from business_logic import urgency_score, root_cause

st.set_page_config(page_title="Customer Complaint Analyzer", layout="wide")
st.title("Customer Complaint Analyzer")

MAX_LEN = 100
NUM_CLASSES = 4

with open("tokenizer.joblib", "rb") as f:
    tokenizer = joblib.load(f)

model = Sequential([
    Embedding(input_dim=10000, output_dim=100),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.3),
    Dense(NUM_CLASSES, activation="softmax")
])
model.build(input_shape=(None, MAX_LEN))
model.load_weights("customer_model-003.keras")

mode = st.radio("Select Analysis Mode", ["Paste Complaint", "Upload CSV"])

if mode == "Paste Complaint":
    st.subheader("Complaint Entry")
    col1, col2 = st.columns(2)

    with col1:
        customer_name = st.text_input("Customer Name")
        product = st.text_input("Product")

    complaint_text = st.text_area("Complaint Text", height=120)

    if st.button("Analyze & Generate CSV"):
        if not customer_name or not product or not complaint_text.strip():
            st.warning("Please fill all fields")
        else:
            seq = tokenizer.texts_to_sequences([complaint_text])
            padded_seq = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
            pred_class = model.predict(padded_seq)
            category = int(pred_class.argmax(axis=1)[0])
            urgency = urgency_score(complaint_text, category)
            root = root_cause(complaint_text)

            result_df = pd.DataFrame([{
                "id": 1,
                "customer_name": customer_name,
                "product": product,
                "complaint": complaint_text,
                "predicted_category": category,
                "urgency": urgency,
                "root_cause": root
            }])

            st.success("Prediction completed")
            st.dataframe(result_df)

            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv, "complaint_prediction.csv", "text/csv")

elif mode == "Upload CSV":
    st.subheader("Complaint Analysis")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())

        text_col = st.selectbox("Select complaint text column", df.columns)
        if st.button("Analysis"):
            with st.spinner("Analyzing complaints..."):
                sequences = tokenizer.texts_to_sequences(df[text_col].astype(str).tolist())
                padded_seq = pad_sequences(sequences, maxlen=MAX_LEN, padding="post")
                preds = model.predict(padded_seq)
                df["predicted_category"] = preds.argmax(axis=1)
                df["urgency"] = df.apply(lambda x: urgency_score(x[text_col], x["predicted_category"]), axis=1)
                df["root_cause"] = df[text_col].apply(root_cause)

            st.success("Analysis completed")
            st.dataframe(df.head(20))
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Result CSV", csv, "complaint_analysis_results.csv", "text/csv")
