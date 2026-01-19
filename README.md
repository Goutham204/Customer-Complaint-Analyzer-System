# Customer Complaint Analyzer System
The Customer Complaint Analyzer System is an NLP-based machine learning project designed to automatically classify and analyze customer complaints. Using historical complaint datasets, this project demonstrates a complete workflow—from text preprocessing and feature engineering to LSTM model training, prediction, and deployment through a user-friendly web interface.

# Project Overview

This project focuses on automating customer complaint analysis using NLP and deep learning (LSTM).
The system enables:

- Complaint categorization
- Determination of urgency
- Identification of root cause

It helps organizations streamline customer support, prioritize critical complaints, and reduce manual analysis effort.

# Features

- End-to-End ML Pipeline: Text preprocessing, tokenization, LSTM model training, evaluation, and deployment
- Complaint Classification: Predicts complaint category using an LSTM-based NLP model
- Urgency & Root Cause Detection: Hybrid approach (rule-based for now) with potential for full ML extension
- Bulk Complaint Analysis: CSV upload mode for multiple complaints
- Real-Time Prediction: Single complaint input through a web interface
- Web Deployment: Streamlit frontend and Flask backend
- Downloadable Results: Predictions saved in CSV with appended urgency and root cause

# Technologies Used

- Programming Language: Python
- Machine Learning / NLP: TensorFlow, Keras, LSTM
- Data Processing: pandas, NumPy, joblib
- Frontend/UI: Streamlit
- Backend/API: Flask
- Visualization: matplotlib, seaborn (for EDA)

# Output

- Real-time complaint classification
- Urgency levels (Low / Medium / High)
- Root cause prediction
- Downloadable CSV with predictions for business analysis
- Interactive web interface for customer support teams

# Ideal Use Cases

- Banks, telecoms, and service industries are managing large volumes of customer complaints
- Customer support teams seeking faster issue triage
- ML practitioners learning NLP and LSTM for business-focused applications
- End-to-end deployment practice of ML projects

License

This project is open-source under the MIT License.
