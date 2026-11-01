"""Streamlit interface using the same raw features as the training pipeline."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from train import FEATURES
from validation import validate_batch
import json

st.set_page_config(page_title="Income Band Classifier", page_icon="💼")
st.markdown("""<style>
.block-container{max-width:1080px;padding-top:3rem}h1,h2,h3{font-family:Georgia,serif!important;letter-spacing:-.035em}h1{font-size:3rem!important}div[data-testid="stForm"]{background:#fffdf7;border:1px solid #d6d8cc;border-radius:18px;padding:26px}div[data-testid="stAlert"]{border-radius:12px}button{border-radius:9px!important}div[data-testid="stFileUploader"]{border:1px dashed #aeb9aa;border-radius:14px;padding:10px}div[data-testid="stDataFrame"]{border-radius:12px;overflow:hidden}@media(max-width:600px){h1{font-size:2.1rem!important}.block-container{padding-top:1.5rem}}
</style>""", unsafe_allow_html=True)
st.title("Income Band Classifier")
st.caption("Educational Adult-dataset demo. Predicts an income band, not an individual salary. Do not use for hiring or pay decisions.")
path = Path(__file__).parent / "data/income-model.joblib"
if not path.exists():
    st.info("Train your local model first: python train.py /path/to/adult.csv")
    st.stop()
# Only load artifacts you created locally with train.py.
model = joblib.load(path)

with st.form("predict"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 90, 30)
        hours = st.slider("Hours per week", 1, 99, 40)
    with col2:
        education = st.selectbox("Education", ["Bachelors", "Masters", "Doctorate", "HS-grad", "Assoc-acdm", "Assoc-voc", "Some-college"])
        occupation = st.selectbox("Occupation", ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"])
    
    if st.form_submit_button("Predict income band"):
        row = pd.DataFrame([{"age": age, "hours-per-week": hours, "education": education, "occupation": occupation}])
        prediction = model.predict(row[FEATURES])[0]
        st.success(f"Predicted Income Band: **{prediction}**")

st.subheader("Batch prediction")
uploaded = st.file_uploader("CSV with age, hours-per-week, education and occupation", type="csv")
if uploaded is not None:
    try:
        frame = pd.read_csv(uploaded)
        frame = validate_batch(frame)
        frame["predicted_income_band"] = model.predict(frame[FEATURES])
        st.dataframe(frame)
        st.download_button("Download predictions", frame.to_csv(index=False), "predictions.csv", "text/csv")
    except ValueError as error:
        st.error(str(error))

st.download_button("Download sample CSV", "age,hours-per-week,education,occupation\n30,40,Bachelors,Tech-support\n", "sample.csv", "text/csv")
metrics_path = path.with_suffix(".metrics.json")
if metrics_path.exists():
    st.subheader("Held-out evaluation")
    metrics = json.loads(metrics_path.read_text())
    st.caption(f"{metrics['holdout_rows']} held-out records. Historical dataset performance is not a guarantee for individuals.")
    st.dataframe(pd.DataFrame(metrics["classification_report"]).T)
