import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ============================
# Load & Prepare Data
# ============================
data = pd.read_csv("spam_ham_dataset.csv (1).zip")

# Convert labels: spam=1, ham=0
data["target"] = data["label"].map({"spam": 1, "ham": 0})

texts = data["text"].astype(str).values
labels = data["target"].values

# Split dataset
X_tr, X_te, y_tr, y_te = train_test_split(
    texts, labels, test_size=0.2, stratify=labels, random_state=10
)

# ============================
# Text Vectorization
# ============================
text_encoder = layers.TextVectorization(
    max_tokens=25000,
    output_mode="tf_idf"
)

text_encoder.adapt(X_tr)

# ============================
# Build Model
# ============================
input_layer = layers.Input(shape=(1,), dtype=tf.string)
encoded = text_encoder(input_layer)

dense1 = layers.Dense(50, activation="relu")(encoded)
dense2 = layers.Dense(25, activation="relu")(dense1)
output_layer = layers.Dense(1, activation="sigmoid")(dense2)

spam_model = models.Model(input_layer, output_layer)

spam_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ============================
# Train
# ============================
spam_model.fit(
    X_tr, y_tr,
    epochs=6,
    batch_size=32,
    validation_split=0.15,
    verbose=2
)

# ============================
# Evaluate
# ============================
pred_probs = spam_model.predict(X_te)
pred_classes = (pred_probs > 0.5).astype(int)

print("Accuracy Score:", accuracy_score(y_te, pred_classes))
print(classification_report(y_te, pred_classes, target_names=["ham", "spam"]))

# ============================
# Test on a new sample
# ============================
test_mail = ["Claim your free reward now! Click the link below!"]
result = spam_model.predict(tf.constant(test_mail))

print("Prediction Score:", result)

if result >= 0.5:
    print("This message is classified as: SPAM")
else:
    print("This message is classified as: NOT SPAM")
