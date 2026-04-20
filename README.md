# Fake Review Detector (BERT)
An AI-powered web application that detects whether a review is **Real or Fake** using a BERT-based NLP model.
## Features
- Transformer-based text classification (BERT)
- Real-time predictions with confidence scores
- Clean and interactive Streamlit UI
## 🛠 Tech Stack
- Python
- Transformers (HuggingFace)
- PyTorch
- Streamlit
- Scikit-learn
##  Run Locally
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
````
## Example
* “BEST PRODUCT EVER!!! MUST BUY!!!” → Fake
* “Battery life is average for this price” → Real
## Project Structure

```
fake_review_detector/
│
├── app.py
├── train_model.py
├── requirements.txt
├── data/
└── .gitignore
```

---
