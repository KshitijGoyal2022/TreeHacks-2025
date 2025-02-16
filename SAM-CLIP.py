import cv2
import torch
import clip
import numpy as np
from PIL import Image

# Import SAM components
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

# ----------- SETUP MODELS ------------

# Choose device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load CLIP (ViT-B/32 model)
model_clip, preprocess_clip = clip.load("ViT-B/32", device=device)

# Load SAM model
# Update this checkpoint path to where you have downloaded the SAM checkpoint file.
sam_checkpoint = "C:/Users/kshit/Downloads/ReactLearn/vlm/sam_vit_b_01ec64.pth"
model_type = "vit_b"  # or "vit_l", etc., depending on your checkpoint
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# Initialize the automatic mask generator from SAM
mask_generator = SamAutomaticMaskGenerator(sam)

# ----------- CAPTURE IMAGE FROM CAMERA ------------

# Initialize your camera (this example uses the default camera)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Press 'c' to capture an image, or 'q' to quit.")
captured_frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    cv2.imshow("Camera Feed", frame)
    key = cv2.waitKey(1)
    if key == ord("c"):
        captured_frame = frame.copy()
        print("Image captured!")
        break
    elif key == ord("q"):
        print("Exiting.")
        cap.release()
        cv2.destroyAllWindows()
        exit()

cap.release()
cv2.destroyAllWindows()

if captured_frame is None:
    print("No image captured. Exiting.")
    exit()

captured_frame_rgb = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB)
pil_image = Image.fromarray(captured_frame_rgb)


print("Running SAM to generate segmentation masks...")
masks = mask_generator.generate(captured_frame_rgb)
print(f"Generated {len(masks)} masks.")

# ----------- USE CLIP TO SCORE EACH MASK ------------

text_prompt = input("Enter a text prompt (e.g., 'a red water bottle'): ").strip()
if text_prompt == "":
    print("No prompt provided. Exiting.")
    exit()

text_input = clip.tokenize([text_prompt]).to(device)

results = []

for mask in masks:
    seg_mask = mask["segmentation"] 
    
    ys, xs = np.where(seg_mask)
    if len(xs) == 0 or len(ys) == 0:
        continue
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    
    if (x2 - x1) < 20 or (y2 - y1) < 20:
        continue

    crop = pil_image.crop((x1, y1, x2, y2))
    
    try:
        image_input = preprocess_clip(crop).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error preprocessing crop: {e}")
        continue

    with torch.no_grad():
        image_feature = model_clip.encode_image(image_input)
        text_feature = model_clip.encode_text(text_input)
    
    image_feature /= image_feature.norm(dim=-1, keepdim=True)
    text_feature /= text_feature.norm(dim=-1, keepdim=True)
    similarity = (image_feature @ text_feature.T).item()
    
    bbox = (x1, y1, x2 - x1, y2 - y1)  # (x, y, width, height)
    results.append((bbox, similarity, crop))

results.sort(key=lambda x: x[1], reverse=True)

if results:
    best_bbox, best_similarity, best_crop = results[0]
    print(f"Best match: Similarity = {best_similarity:.4f}")
    
    output_img = captured_frame_rgb.copy()
    x, y, w, h = best_bbox
    cv2.rectangle(output_img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
    
    cv2.imshow("Best Match", cv2.cvtColor(np.array(output_img), cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No valid segmented object found matching the text prompt.")
