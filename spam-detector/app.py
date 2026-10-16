import streamlit as st

from model import predict, train_model


@st.cache_resource
def load_model():
    """Train once and reuse the model on every interaction."""
    return train_model()


st.title("Spam Message Detector")
st.write("Type or paste a message below and click **Check Message**.")

model = load_model()

message = st.text_area("Your message", height=150, placeholder="e.g. You won a free prize! Click here to claim")

if st.button("Check Message"):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        label, spam_probability = predict(model, message)
        if label == "SPAM":
            st.error(f"🚫 SPAM  (spam probability: {spam_probability:.0%})")
        else:
            st.success(f"✅ NOT SPAM  (spam probability: {spam_probability:.0%})")

st.caption("Trained on a very small example dataset, so results are for learning purposes only.")
