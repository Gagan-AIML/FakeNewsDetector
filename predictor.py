import json
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# Load model
model = load_model("fake_news_model.h5")

# Load tokenizer
with open("tokenizer.json") as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

# Clean + predict (simple version without spaCy)
def predict_fake_news(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=200)
    prediction = model.predict(padded)[0][0]
    return "Real" if prediction >= 0.5 else "Fake"
