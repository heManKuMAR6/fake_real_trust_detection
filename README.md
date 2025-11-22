# Fake News Detection Project

This project implements three ML/NLP approaches for Fake News Detection:
1. **TF-IDF + Logistic Regression**
2. **LSTM Model**
3. **DistilBERT Transformer Model**

A Streamlit app is included to allow real-time predictions.

## How to Run
Activate venv:
```
.env\Scriptsctivate
```

Install deps:
```
pip install -r requirements.txt
```

Train DistilBERT:
```
python model3_distilbert.py
```

Run app:
```
streamlit run app.py
```
