import numpy as np
import cv2, os
from PIL import Image
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.applications.resnet50 import preprocess_input as res_pre, decode_predictions
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_pre

IMG_SIZE = (224, 224)

# ---------------------------------------------------
# Load ImageNet Pretrained Models
# ---------------------------------------------------
model_rn = ResNet50(include_top=True, weights="imagenet")
model_en = EfficientNetB0(include_top=True, weights="imagenet")


# ---------------------------------------------------
# Augment: Test-Time Augmentation (TTA)
# ---------------------------------------------------
def generate_crops(pil_img):
    width, height = pil_img.size
    w0, h0 = IMG_SIZE

    # Resize if too small
    if width < w0 or height < h0:
        new_w = max(width, w0)
        new_h = max(height, h0)
        pil_img = pil_img.resize((new_w, new_h))
        width, height = new_w, new_h

    # 5 crop positions
    crop_list = [
        pil_img.crop(((width - w0) // 2, (height - h0) // 2, (width + w0) // 2, (height + h0) // 2)),  # center
        pil_img.crop((0, 0, w0, h0)),                                       # top-left
        pil_img.crop((width - w0, 0, width, h0)),                           # top-right
        pil_img.crop((0, height - h0, w0, height)),                         # bottom-left
        pil_img.crop((width - w0, height - h0, width, height))              # bottom-right
    ]

    # Add horizontal flips
    flip_crops = [x.transpose(Image.FLIP_LEFT_RIGHT) for x in crop_list]

    return crop_list + flip_crops


# ---------------------------------------------------
# Prediction Using Ensemble + TTA
# ---------------------------------------------------
def classify_image(img_path):
    pil_im = Image.open(img_path).convert("RGB")
    all_views = generate_crops(pil_im)

    preds_rn, preds_en = [], []

    for view in all_views:
        arr = cv2.resize(np.array(view), IMG_SIZE).astype("float32")

        # Expand dims: (H,W,C) → (1,H,W,C)
        rn_out = model_rn.predict(res_pre(arr)[None], verbose=0)[0]
        en_out = model_en.predict(eff_pre(arr)[None], verbose=0)[0]

        preds_rn.append(rn_out)
        preds_en.append(en_out)

    # Average ensemble over all crops and both models
    avg_scores = (np.mean(preds_rn, axis=0) + np.mean(preds_en, axis=0)) / 2.0

    # Get best class index
    best_id = int(np.argmax(avg_scores))

    # Decode that single class to human-readable label
    one_hot = np.zeros((1, 1000))
    one_hot[0, best_id] = 1
    label = decode_predictions(one_hot, top=1)[0][0][1]

    print("\n==============================")
    print("   Final Prediction Result")
    print("==============================\n")
    print(f"Predicted object: {label}")
    print(f"Confidence score (approx): {avg_scores[best_id]:.4f}")


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
if __name__ == "__main__":
    img_path = input("Enter path to image: ").strip()

    if os.path.isfile(img_path):
        classify_image(img_path)
    else:
        print("❌ Error: File not found!")
