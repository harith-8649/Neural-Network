# two_digit_image_recognition.py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

# ------------------------------------------------
# 1. Load MNIST Dataset
# ------------------------------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize + Expand dims
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

x_train = x_train[..., np.newaxis]   # (28,28,1)
x_test  = x_test[..., np.newaxis]

# ------------------------------------------------
# 2. Build digit classifier (0–9)
# ------------------------------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation="relu", input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPool2D(),

    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPool2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")   # 10 digits output
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

print("Training model...")
model.fit(x_train, y_train, epochs=3, batch_size=64, validation_split=0.1, verbose=2)

# ------------------------------------------------
# 3. Load Two-Digit User Image
# ------------------------------------------------
def load_two_digit_image(path):
    """
    Input image should contain two digits side by side, e.g., '25'
    """
    img = Image.open(path).convert("L")   # grayscale
    img = img.resize((56, 28))            # width = 28*2 → for 2 digits
    arr = np.array(img).astype("float32") / 255.0

    # Split into left(28x28) and right(28x28)
    left_digit  = arr[:, :28]
    right_digit = arr[:, 28:]

    return left_digit[..., np.newaxis], right_digit[..., np.newaxis]

# ------------------------------------------------
# 4. Predict Two Digits
# ------------------------------------------------
def predict_two_digit(path):
    left, right = load_two_digit_image(path)

    # Predict individually
    pred_left  = np.argmax(model.predict(left.reshape(1,28,28,1), verbose=0))
    pred_right = np.argmax(model.predict(right.reshape(1,28,28,1), verbose=0))

    result = int(f"{pred_left}{pred_right}")

    print("\n===============================")
    print(f"Image: {path}")
    print(f"Predicted Number : {result}")
    print("===============================\n")

    # Show the image
    full_img = np.concatenate([left.squeeze(), right.squeeze()], axis=1)
    plt.imshow(full_img, cmap="gray")
    plt.title(f"Prediction: {result}")
    plt.axis("off")
    plt.show()

# ------------------------------------------------
# 5. Ask for image
# ------------------------------------------------
path = input("\nEnter 2-digit image path: ")
predict_two_digit(path)
