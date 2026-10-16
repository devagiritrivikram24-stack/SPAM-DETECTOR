# Spam Message Detector

A minimal web app that classifies a message as **SPAM** or **NOT SPAM**, built with Streamlit and scikit-learn.

## Files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit web interface (text box + "Check Message" button) |
| `model.py` | Trains the classifier and makes predictions |
| `spam_data.csv` | Small labeled dataset (`label`, `text`) |
| `requirements.txt` | Python dependencies |

## Setup

You need Python 3.9 or newer.

1. (Optional but recommended) create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:

   ```bash
   streamlit run app.py
   ```

4. Open the URL shown in the terminal (usually http://localhost:8501), type a message, and click **Check Message**.

## How it works

1. `spam_data.csv` holds example messages labeled `spam` or `ham` (not spam).
2. `model.py` converts the text into numbers with **TF-IDF** and trains a **Multinomial Naive Bayes** classifier.
3. When you click the button, the app predicts the label and shows the spam probability.

The model trains in a fraction of a second each time the app starts, so no saved model file is needed.

## Improving it

- Add more rows to `spam_data.csv` (the more varied examples, the better).
- Try a different classifier, e.g. `LogisticRegression`, in `model.py`.

Note: the bundled dataset is tiny and meant for learning. It will make mistakes on real-world messages.
