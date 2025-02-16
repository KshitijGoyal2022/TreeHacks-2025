import cv2
import torch
import clip
from PIL import Image
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model, preprocess = clip.load("ViT-B/32", device=device)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'c' to capture an image, or 'q' to quit.")
captured_frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    cv2.imshow("Webcam", frame)
    
    key = cv2.waitKey(1)
    if key == ord('c'): 
        captured_frame = frame.copy()
        print("Image captured!")
        break
    elif key == ord('q'):
        print("Quitting.")
        break

cap.release()
cv2.destroyAllWindows()

if captured_frame is None:
    print("No image captured. Exiting.")
    exit()

captured_frame_rgb = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB)
image = Image.fromarray(captured_frame_rgb)

image_input = preprocess(image).unsqueeze(0).to(device)

text_prompts = ["a photo of a cat", "a photo of a dog", "a diagram", "a water bottle", "a photo of a human"]
text_inputs = clip.tokenize(text_prompts).to(device)

with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)

image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)

similarities = (image_features @ text_features.T).squeeze(0).cpu().numpy()

print("Similarity scores:")
for prompt, score in zip(text_prompts, similarities):
    print(f'  "{prompt}": {score:.4f}')
