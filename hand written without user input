import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, datasets

# ====================== DATA PREPARATION ======================

def get_data():
    # Load MNIST directly from Keras
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

    # Normalize to [0, 1]
    train_images = train_images.astype("float32") / 255.0
    test_images  = test_images.astype("float32") / 255.0

    # Add channel dimension (28, 28) -> (28, 28, 1)
    train_images = np.expand_dims(train_images, axis=-1)
    test_images  = np.expand_dims(test_images, axis=-1)

    return (train_images, train_labels), (test_images, test_labels)


# ====================== MODEL CREATION ======================

def create_cnn_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ====================== TRAINING FUNCTION ======================

def run_training(model, x_train, y_train, num_epochs=5, batch_sz=128):
    history = model.fit(
        x_train,
        y_train,
        epochs=num_epochs,
        batch_size=batch_sz,
        validation_split=0.1,
        verbose=2
    )
    return history


# ====================== EVALUATION & DEMO ======================

def evaluate_model(model, x_test, y_test, samples_to_show=10):
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nFinal Test Accuracy: {acc:.4f}")
    print(f"Final Test Loss:     {loss:.4f}")

    # Predict on some sample images
    sample_images = x_test[:samples_to_show]
    sample_labels = y_test[:samples_to_show]

    predictions = model.predict(sample_images, verbose=0)
    predicted_digits = np.argmax(predictions, axis=1)

    print("\nTrue labels:     ", sample_labels)
    print("Predicted labels:", predicted_digits)


# ====================== MAIN ENTRY POINT ======================

def run():
    (x_train, y_train), (x_test, y_test) = get_data()
    cnn_model = create_cnn_model()

    print("\nModel Summary:")
    cnn_model.summary()

    run_training(cnn_model, x_train, y_train, num_epochs=5, batch_sz=128)
    evaluate_model(cnn_model, x_test, y_test, samples_to_show=10)


if __name__ == "__main__":
    run()
