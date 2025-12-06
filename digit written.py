# two_digit_image_recognition.py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

# -----------------------------
# 1. Choose digits
# -----------------------------
DIGIT_A = 2       # label 0
DIGIT_B = 7       # label 1

# -----------------------------
# 2. Load and filter MNIST
# -----------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

def filter_digits(x, y, a, b):
    mask = (y == a) | (y == b)
    x = x[mask].astype("float32") / 255.0
    y = (y[mask] == b).astype("int32")      # DIGIT_B → 1
    return x[..., np.newaxis], y

x_train_f, y_train_f = filter_digits(x_train, y_train, DIGIT_A, DIGIT_B)
x_test_f,  y_test_f  = filter_digits(x_test,  y_test,  DIGIT_A, DIGIT_B)

# -----------------------------
# 3. Build model
# -----------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, 3, activation="relu", input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# -----------------------------
# 4. Train
# -----------------------------
print("Training model...")
model.fit(x_train_f, y_train_f, validation_split=0.1, epochs=5, batch_size=64, verbose=2)

# -----------------------------
# 5. Function to load user image
# -----------------------------
def load_image(path):
    img = Image.open(path).convert("L")        # grayscale
    img = img.resize((28,28))                  # MNIST size
    arr = np.array(img).astype("float32") / 255.0
    return arr[..., np.newaxis]               # (28,28,1)

# -----------------------------
# 6. Predict user image
# -----------------------------
def predict_image(path):
    img = load_image(path)
    prob = float(model.predict(img.reshape(1,28,28,1), verbose=0)[0][0])
    pred = DIGIT_B if prob >= 0.5 else DIGIT_A

    print("\n====================")
    print(f"Image: {path}")
    print(f"Prediction : {pred}")
    print(f"Confidence : {prob:.4f}")
    print("====================\n")

    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(f"Pred: {pred}  (Score: {prob:.3f})")
    plt.axis("off")
    plt.show()

# -----------------------------
# 7. Ask user for image path
# -----------------------------
path = input("\nEnter image path for recognition: ")
predict_image(path)
