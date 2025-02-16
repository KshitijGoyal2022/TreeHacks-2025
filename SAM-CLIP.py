import cv2
import torch
import clip
import numpy as np
from PIL import Image
import pyrealsense2 as rs
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_clip, preprocess_clip = clip.load("ViT-B/32", device=device)

sam_checkpoint = "C:/Users/kshit/Downloads/ReactLearn/vlm/sam_vit_b_01ec64.pth"
model_type = "vit_b"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)

pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

print("Press 'c' to capture an image, or 'q' to quit.")
captured_frame = None

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        images = np.hstack((color_image, depth_colormap))

        cv2.imshow('RealSense', images)
        key = cv2.waitKey(1)

        if key == ord('c'):
            captured_frame = color_image.copy()
            print("Image captured!")
            break
        elif key == ord('q'):
            print("Exiting.")
            cv2.destroyAllWindows()
            exit()

    if captured_frame is None:
        print("No image captured. Exiting.")
        exit()

    captured_frame_rgb = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(captured_frame_rgb)

    print("Running SAM to generate segmentation masks...")
    masks = mask_generator.generate(captured_frame_rgb)
    print(f"Generated {len(masks)} masks.")

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
        
        bbox = (x1, y1, x2 - x1, y2 - y1)
        results.append((bbox, similarity, crop))

    results.sort(key=lambda x: x[1], reverse=True)

    if results:
        best_bbox, best_similarity, best_crop = results[0]
        print(f"Best match: Similarity = {best_similarity:.4f}")
        
        output_img = captured_frame_rgb.copy()
        x, y, w, h = best_bbox
        cv2.rectangle(output_img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        
        depth_colormap_resized = cv2.resize(depth_colormap, (output_img.shape[1], output_img.shape[0]))
        final_output = np.hstack((cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR), depth_colormap_resized))
        
        cv2.imshow("Results (Color + Depth)", final_output)
        cv2.waitKey(0)
    else:
        print("No valid segmented object found matching the text prompt.")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
