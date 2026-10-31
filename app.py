"""Streamlit interface using the same raw features as the training pipeline."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from train import FEATURES

st.set_page_config(page_title="Income Band Classifier", page_icon="💼")
st.title("Income Band Classifier")
st.caption("Educational Adult-dataset demo. Predicts an income band, not an individual salary. Do not use for hiring or pay decisions.")
path = Path(__file__).parent / "data/income-model.joblib"
if not path.exists():
    st.info("Train your local model first: python train.py /path/to/adult.csv")
    st.stop()
# Only load artifacts you created locally with train.py.
model = joblib.load(path)
with st.form("predict"):
    age = st.slider("Age", 18, 90, 30)
    hours = st.slider("Hours per week", 1, 99, 40)
    education = st.selectbox("Education", ["Bachelors", "Masters", "Doctorate", "HS-grad", "Assoc-acdm", "Assoc-voc", "Some-college"])
    occupation = st.selectbox("Occupation", ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"])
    if st.form_submit_button("Predict income band"):
        row = pd.DataFrame([{"age": age, "hours-per-week": hours, "education": education, "occupation": occupation}])
        st.success(str(model.predict(row[FEATURES])[0]))
st.subheader("Batch prediction")
uploaded = st.file_uploader("CSV with age, hours-per-week, education and occupation", type="csv")
if uploaded is not None:
    try:
        frame = pd.read_csv(uploaded)
        missing = set(FEATURES) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
        for column in ["age", "hours-per-week"]:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        frame["predicted_income_band"] = model.predict(frame[FEATURES])
        st.dataframe(frame)
        st.download_button("Download predictions", frame.to_csv(index=False), "predictions.csv", "text/csv")
    except ValueError as error:
        st.error(str(error))
