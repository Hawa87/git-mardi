import cv2
import torch
from torchvision import models, transforms
from torchvision.utils import draw_bounding_boxes
from PIL import Image
import numpy as np

# Define classes
classes = ['__background__', 'Apple', 'Banana', 'Orange']
num_classes = len(classes)

# Load the trained model
def load_trained_model(model_path, num_classes):
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

# Predict function
def predict(image_tensor, model, device):
    model.to(device)
    image_tensor = image_tensor.to(device)
    model.eval()
    with torch.no_grad():
        prediction = model([image_tensor])
    return prediction

# Draw predictions on the image
def draw_predictions(image_tensor, prediction, threshold=0.5):
    boxes = prediction[0]['boxes']
    scores = prediction[0]['scores']
    labels = prediction[0]['labels']

    keep = scores >= threshold
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    label_names = [classes[i] for i in labels]
    captions = [f"{name}: {score:.2f}" for name, score in zip(label_names, scores)]
    drawn = draw_bounding_boxes((image_tensor * 255).byte(), boxes, captions, width=2)
    return drawn

# Transformation to tensor
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load model and set device
model_path = 'fruit_detect.pth'  # 👈 Replace with your model file
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_trained_model(model_path, num_classes)

# Live webcam detection
def live_detect_from_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access webcam.")
        return

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL then to tensor
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        image_tensor = transform(pil_img)

        # Get prediction
        prediction = predict(image_tensor, model, device)
        drawn_tensor = draw_predictions(image_tensor, prediction)

        # Convert tensor to numpy for OpenCV display
        drawn_image = drawn_tensor.permute(1, 2, 0).cpu().numpy()
        drawn_image_bgr = cv2.cvtColor(drawn_image, cv2.COLOR_RGB2BGR)

        # Display
        cv2.imshow("Fruit Detection", drawn_image_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run
live_detect_from_webcam()
