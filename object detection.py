import torch
import torchvision
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np

# Default COCO labels (used only if model metadata isn't available)
COCO_LABELS = [
    "_background_", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter",
    "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "N/A", "backpack", "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "N/A",
    "dining table", "N/A", "N/A", "toilet", "N/A", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "N/A", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# ---------------------------
#   LOAD MODEL + LABELS
# ---------------------------
try:
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
    
    model_weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    detector = fasterrcnn_resnet50_fpn(weights=model_weights)
    LABELS = model_weights.meta.get("categories", COCO_LABELS)

except Exception:
    detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    LABELS = COCO_LABELS

device = "cuda" if torch.cuda.is_available() else "cpu"
detector.to(device).eval()


# ---------------------------
#   DETECTION FUNCTION
# ---------------------------
def run_detection(image_path, threshold=0.60, output_file="detected_output.jpg"):
    # Convert image to Tensor
    img = Image.open(image_path).convert("RGB")
    tensor_img = transforms.ToTensor()(img).to(device)

    # Run inference
    with torch.no_grad():
        prediction = detector([tensor_img])[0]

    boxes = prediction["boxes"].cpu().numpy()
    labels = prediction["labels"].cpu().numpy()
    scores = prediction["scores"].cpu().numpy()

    # Prepare OpenCV image
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Draw results
    for box, cls, conf in zip(boxes, labels, scores):
        if conf < threshold:
            continue

        x1, y1, x2, y2 = map(int, box)
        name = LABELS[cls] if cls < len(LABELS) else f"class_{cls}"

        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_cv, f"{name} {conf:.2f}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imwrite(output_file, img_cv)
    print(f"✔ Output saved as: {output_file}")

    return boxes, labels, scores


# ---------------------------
#   MAIN INPUT + PRINTING
# ---------------------------
image_file = input("Enter image path: ")

det_boxes, det_labels, det_scores = run_detection(image_file)

for b, lbl, sc in zip(det_boxes, det_labels, det_scores):
    if sc >= 0.60:
        obj_name = LABELS[lbl] if lbl < len(LABELS) else str(lbl)
        print(f"Detected: {obj_name} | ID: {lbl} | Score: {sc:.3f} | Box: {b.astype(int).tolist()}")
