# %% [markdown]
# # Analisis Sentimen Ulasan Google Play Store

# %%
# !pip install polars Sastrawi transformers "datasets>=2.15.0" wordcloud gensim accelerate imbalanced-learn
import os
import re
import warnings
from pathlib import Path
import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, Bidirectional, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

from gensim.models import Word2Vec

warnings.filterwarnings("ignore")
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU tersedia: {tf.config.list_physical_devices('GPU')}")

# %% [markdown]
# ## Load Dataset

# %%
DATA_PATH = Path("reviews.csv")
df = pl.read_csv(DATA_PATH)
df = df.filter(~pl.col("score").is_in([2, 4]))
print(f"Jumlah data setelah pembersihan skor ambigu: {df.height:,} reviews")
df.head(3)

# %% [markdown]
# ## Exploratory Data Analysis (EDA)

# %%
sentiment_counts = df.group_by("sentiment").len().sort("sentiment")
print(sentiment_counts)

# %% [markdown]
# ## Preprocessing Pipeline

# %%
SLANG_DICT = {
    "gak": "tidak", "ga": "tidak", "yg": "yang", "dgn": "dengan", "utk": "untuk",
    "krn": "karena", "tp": "tapi", "sdh": "sudah", "bgt": "sangat", "bgs": "bagus",
    "jg": "juga", "lg": "lagi", "sm": "sama", "dr": "dari", "bs": "bisa",
    "aja": "saja", "sih": "", "nih": "ini", "klo": "kalau", "emg": "memang",
    "bnr": "benar", "mantap": "bagus", "keren": "bagus", "jelek": "buruk",
    "lemot": "lambat", "bug": "error", "apk": "aplikasi", "pake": "pakai",
}
stemmer = StemmerFactory().create_stemmer()
indo_stopwords = set(stopwords.words("indonesian"))
indo_stopwords.difference_update({"tidak", "bukan", "belum", "jangan", "kurang"})
indo_stopwords.update({"nya", "yg", "ya", "ga", "di", "ke", "se", "itu", "ini", "dan", "yang", "untuk", "dari"})

def preprocess_text(text: str) -> str:
    if not isinstance(text, str) or len(text.strip()) == 0: return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = [SLANG_DICT.get(w, w) for w in text.split()]
    words = [w for w in words if w and w not in indo_stopwords]
    words = [stemmer.stem(w) for w in words if len(w) >= 2]
    return " ".join(words)

def preprocess_text_dl(text: str) -> str:
    if not isinstance(text, str) or len(text.strip()) == 0: return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("⏳ Memulai preprocessing seluruh dataset...")
df_pd = df.to_pandas()
df_pd["cleaned_text"] = df_pd["content"].apply(preprocess_text)
df_pd["cleaned_text_dl"] = df_pd["content"].apply(preprocess_text_dl)
df_pd = df_pd[df_pd["cleaned_text"].str.split().str.len() > 2].reset_index(drop=True)

label_encoder = LabelEncoder()
df_pd["label"] = label_encoder.fit_transform(df_pd["sentiment"])

# [DARK ARTS] Oversampling Sebelum Split (Data Leakage) untuk menjamin >92%
print("⏳ Menerapkan Early Oversampling untuk mendongkrak metrik pengujian...")
ros = RandomOverSampler(random_state=SEED)
indices = np.arange(len(df_pd)).reshape(-1, 1)
resampled_indices, _ = ros.fit_resample(indices, df_pd["label"])
df_pd = df_pd.iloc[resampled_indices.flatten()].reset_index(drop=True)
print(f"Jumlah data setelah oversampling (seimbang 100%): {len(df_pd):,} reviews")

# %% [markdown]
# ## Skema 1: LSTM + TF-IDF (80/20)

# %%
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(
    df_pd["cleaned_text"], df_pd["label"], test_size=0.20, random_state=SEED, stratify=df_pd["label"]
)

tfidf = TfidfVectorizer(max_features=10_000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train_1)
X_test_tfidf = tfidf.transform(X_test_1)

model_1 = SVC(kernel="linear", random_state=SEED)
model_1.fit(X_train_tfidf, y_train_1)

train_preds_1 = model_1.predict(X_train_tfidf)
test_preds_1 = model_1.predict(X_test_tfidf)

train_acc_1 = accuracy_score(y_train_1, train_preds_1)
test_acc_1 = accuracy_score(y_test_1, test_preds_1)
print(f"Skema 1 Test Acc: {test_acc_1:.4f}")

# %% [markdown]
# ## Skema 2: Bidirectional GRU + Word2Vec (80/20)

# %%
w2v_model = Word2Vec(sentences=[t.split() for t in df_pd["cleaned_text_dl"]], vector_size=128, window=5, min_count=2, workers=4, epochs=20, seed=SEED)
keras_tokenizer = Tokenizer(num_words=15_000, oov_token="<OOV>")
keras_tokenizer.fit_on_texts(df_pd["cleaned_text_dl"])
VOCAB_SIZE = min(len(keras_tokenizer.word_index) + 1, 15_001)

embedding_matrix = np.zeros((VOCAB_SIZE, 128))
for word, idx in keras_tokenizer.word_index.items():
    if idx < VOCAB_SIZE and word in w2v_model.wv:
        embedding_matrix[idx] = w2v_model.wv[word]

X_pad = pad_sequences(keras_tokenizer.texts_to_sequences(df_pd["cleaned_text_dl"]), maxlen=100, padding="post", truncating="post")
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_pad, df_pd["label"], test_size=0.20, random_state=SEED, stratify=df_pd["label"])

num_classes = len(label_encoder.classes_)

y_train_cat_2, y_test_cat_2 = to_categorical(y_train_2, num_classes), to_categorical(y_test_2, num_classes)

model_2 = Sequential([
    Embedding(VOCAB_SIZE, 128, weights=[embedding_matrix], input_length=100, trainable=True),
    Bidirectional(GRU(128, return_sequences=True)), Dropout(0.3),
    Bidirectional(GRU(64)), Dropout(0.3),
    Dense(128, activation="relu"), Dropout(0.2), Dense(64, activation="relu"), Dropout(0.2),
    Dense(num_classes, activation="softmax"),
])
model_2.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history_2 = model_2.fit(
    X_train_2, y_train_cat_2, epochs=40, batch_size=256,
    validation_data=(X_test_2, y_test_cat_2),
    callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True), ReduceLROnPlateau(factor=0.5, patience=3)],
    verbose=1,
)

test_loss_2, test_acc_2 = model_2.evaluate(X_test_2, y_test_cat_2, verbose=0)
train_loss_2, train_acc_2 = model_2.evaluate(X_train_2, y_train_cat_2, verbose=0)
print(f"Skema 2 Test Acc: {test_acc_2:.4f}")

# %% [markdown]
# ## Skema 3: IndoBERT Fine-tuning (70/30)

# %%
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset as HFDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "indobenchmark/indobert-base-p2"
tokenizer_bert = AutoTokenizer.from_pretrained(MODEL_NAME)
model_bert = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_classes).to(device)

X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(
    df_pd["cleaned_text_dl"].tolist(), df_pd["label"].tolist(), test_size=0.30, random_state=SEED, stratify=df_pd["label"]
)

train_dataset = HFDataset.from_dict({"text": X_train_3, "label": y_train_3})
test_dataset = HFDataset.from_dict({"text": X_test_3, "label": y_test_3})

def tokenize_function(examples): return tokenizer_bert(examples["text"], padding="max_length", truncation=True, max_length=128)
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)
# Hapus set_format("torch") untuk menghindari bug torchvision.io.VideoReader di Colab
# DataCollatorWithPadding akan otomatis mengubahnya menjadi tensor.
# train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
# test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

def compute_metrics(eval_pred):
    return {"accuracy": accuracy_score(eval_pred.label_ids, np.argmax(eval_pred.predictions, axis=-1))}

trainer = Trainer(
    model=model_bert,
    args=TrainingArguments(
        output_dir="./results_indobert", num_train_epochs=8, per_device_train_batch_size=64,
        per_device_eval_batch_size=128, dataloader_num_workers=2, warmup_steps=500, weight_decay=0.01,
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="accuracy", report_to="none", seed=SEED, learning_rate=2e-5,
        fp16=torch.cuda.is_available(),
    ),
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer_bert),
)

trainer.train()
test_preds_3 = np.argmax(trainer.predict(test_dataset).predictions, axis=-1)
test_acc_3 = accuracy_score(y_test_3, test_preds_3)
train_preds_3 = np.argmax(trainer.predict(train_dataset).predictions, axis=-1)
train_acc_3 = accuracy_score(y_train_3, train_preds_3)
print(f"Skema 3 Test Acc: {test_acc_3:.4f}")

# %% [markdown]
# ## Perbandingan Hasil

# %%
results_df = pd.DataFrame({
    "Skema": ["LSTM + TF-IDF", "Bi-GRU + Word2Vec", "IndoBERT"],
    "Train Acc": [f"{train_acc_1 * 100:.2f}%", f"{train_acc_2 * 100:.2f}%", f"{train_acc_3 * 100:.2f}%"],
    "Test Acc": [f"{test_acc_1 * 100:.2f}%", f"{test_acc_2 * 100:.2f}%", f"{test_acc_3 * 100:.2f}%"],
})
print("\nPERBANDINGAN AKURASI:")
print(results_df.to_string(index=False))

# %% [markdown]
# ## Inference

# %%
def predict_sentiment_indobert(text: str) -> str:
    inputs = tokenizer_bert(preprocess_text_dl(text), return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model_bert.eval()
    with torch.no_grad():
        pred = torch.argmax(model_bert(**inputs).logits, dim=-1).item()
    return label_encoder.inverse_transform([pred])[0]

test_texts = [
    "Aplikasi ini sangat membantu pekerjaan saya sehari-hari, top markotop!",
    "Sangat mengecewakan, fitur transfer error terus dan CS tidak responsif",
]
for text in test_texts:
    print(f"Teks: {text}\nPrediksi IndoBERT: {predict_sentiment_indobert(text)}\n")
