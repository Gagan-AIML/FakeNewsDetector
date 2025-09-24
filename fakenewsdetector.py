import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import spacy


# 1. Data Loading & Preprocessing
df = pd.read_csv('/content/Fake_Real_Data.csv')

# Check basic dataset info
print("Dataset shape:", df.shape)
print("Missing values:", df.isna().sum().sum())
print("Duplicates:", df.duplicated().sum())
print("Class distribution:\n", df['label'].value_counts())

# Encode labels (Fake → 0, Real → 1)
df['label'] = df['label'].map({'Fake': 0, 'Real': 1})


# 2. Tokenization & Padding
MAX_VOCAB = 5000
MAX_LEN = 200

tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token='<OOV>')
tokenizer.fit_on_texts(df['Text'])

sequences = tokenizer.texts_to_sequences(df['Text'])
X = pad_sequences(sequences, maxlen=MAX_LEN)
y = np.array(df['label'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vocab_size = min(MAX_VOCAB, len(tokenizer.word_index) + 1)


# 3. Model Architecture
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=64, input_length=MAX_LEN),
    LSTM(32, return_sequences=True),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(model.summary())


# 4. Training
history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=64,
    validation_data=(X_test, y_test),
    verbose=1
)

# Load spaCy for preprocessing
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    """Clean and lemmatize text using spaCy."""
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)

def predict_fake_news(news_text):
    """Predict whether a news article is Fake or Real."""
    cleaned_text = preprocess_text(news_text)
    seq = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    prediction = model.predict(padded)[0][0]
    return "Real" if prediction >= 0.5 else "Fake"


# 5. Sample Predictions
sample1 = """At least four persons were killed and dozens feared washed away in Uttarkashi district of Uttarakhand after flash floods triggered by torrential rain hit the Kheer Ganga river on Tuesday afternoon."""
sample2 = """NASA confirms water found on the Moon's surface again!"""

print("Sample 1:", predict_fake_news(sample1))
print("Sample 2:", predict_fake_news(sample2))
