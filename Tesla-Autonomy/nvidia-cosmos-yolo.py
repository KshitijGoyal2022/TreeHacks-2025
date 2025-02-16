import os
from dotenv import load_dotenv
load_dotenv()

!pip install -q kaggle
from google.colab import files

# Upload your kaggle.json file (only required once)
files.upload()  # select your kaggle.json

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json


# Download the competition dataset (this will produce a file named "tesla-real-world-video-q-a.zip")
!kaggle competitions download -c tesla-real-world-video-q-a

# Unzip the downloaded file into a folder (e.g. "tesla_dataset")
!unzip -q tesla-real-world-video-q-a.zip -d tesla_dataset

# === 2. Install Required Libraries ===
!pip install ultralytics openai torch torchvision huggingface_hub peft Pillow
!pip install transformers==4.27.0

# Download the pre-trained YOLOv8 nano model weights if not already available
!wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O yolov8n.pt

# Load the model in your Python code
from ultralytics import YOLO
model = YOLO("yolov8n.pt")  # This loads the model with the downloaded weights

import os
import cv2
import uuid
import json
import csv
import re
import requests
from ultralytics import YOLO
import openai

# ================= NVIDIA Cosmos-Nemotron-34B VLM Configuration =================
NVLM_INVOKE_URL = "https://ai.api.nvidia.com/v1/vlm/nvidia/cosmos-nemotron-34b"
NVLM_ASSET_URL = "https://api.nvcf.nvidia.com/v2/nvcf/assets"
NVLM_SUPPORTED = {
    "png": ["image/png", "img"],
    "jpg": ["image/jpg", "img"],
    "jpeg": ["image/jpeg", "img"],
    "mp4": ["video/mp4", "video"],
}

TEST_NVCF_API_KEY = os.getenv('TEST_NVCF_API_KEY')
if not TEST_NVCF_API_KEY:
    raise ValueError("Please set the TEST_NVCF_API_KEY environment variable.")

# ================= OpenAI ChatGPT Configuration =================
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")
openai.api_key = OPENAI_API_KEY

# ================= Helper Functions for NVIDIA VLM =================
def get_extension(filename):
    _, ext = os.path.splitext(filename)
    return ext[1:].lower()

def mime_type(ext):
    return NVLM_SUPPORTED[ext][0]

def media_type(ext):
    return NVLM_SUPPORTED[ext][1]

def upload_asset(media_file, description):
    ext = get_extension(media_file)
    if ext not in NVLM_SUPPORTED:
        raise ValueError(f"Unsupported file format: {ext}")
    with open(media_file, "rb") as f:
        data_input = f.read()
    headers = {
        "Authorization": f"Bearer {TEST_NVCF_API_KEY}",
        "Content-Type": "application/json",
        "accept": "application/json",
    }
    response = requests.post(
        NVLM_ASSET_URL,
        headers=headers,
        json={"contentType": mime_type(ext), "description": description},
        timeout=30,
    )
    response.raise_for_status()
    res_json = response.json()
    upload_url = res_json["uploadUrl"]
    asset_id = res_json["assetId"]
    upload_headers = {
        "x-amz-meta-nvcf-asset-description": description,
        "content-type": mime_type(ext),
    }
    put_response = requests.put(
        upload_url,
        data=data_input,
        headers=upload_headers,
        timeout=300,
    )
    put_response.raise_for_status()
    if put_response.status_code == 200:
        print(f"Uploaded asset_id {asset_id} successfully!")
    else:
        print(f"Upload failed for asset_id {asset_id}.")
    return str(uuid.UUID(asset_id))

def delete_asset(asset_id):
    headers = {
        "Authorization": f"Bearer {TEST_NVCF_API_KEY}",
    }
    url = f"{NVLM_ASSET_URL}/{asset_id}"
    response = requests.delete(url, headers=headers, timeout=30)
    response.raise_for_status()

def get_scene_description(media_file, query="Describe the scene", stream=False):
    """
    Uploads the media file to NVIDIA's asset service, queries the Cosmos-Nemotron-34B VLM,
    and returns a scene description.
    """
    ext = get_extension(media_file)
    if ext not in NVLM_SUPPORTED:
        raise ValueError(f"Unsupported file format: {ext}")

    asset_id = upload_asset(media_file, "Reference media file")
    media_content = f'<{media_type(ext)} src="data:{mime_type(ext)};asset_id,{asset_id}" />'
    asset_seq = asset_id  # Only one asset is used.
    headers = {
        "Authorization": f"Bearer {TEST_NVCF_API_KEY}",
        "Content-Type": "application/json",
        "NVCF-INPUT-ASSET-REFERENCES": asset_seq,
        "NVCF-FUNCTION-ASSET-IDS": asset_seq,
        "Accept": "application/json",
    }
    if stream:
        headers["Accept"] = "text/event-stream"

    messages = [{
        "role": "user",
        "content": f"{query} {media_content}",
    }]
    payload = {
        "max_tokens": 1024,
        "temperature": 0.2,
        "top_p": 0.7,
        "seed": 50,
        "num_frames_per_inference": 8,
        "messages": messages,
        "stream": stream,
        "model": "nvidia/vila",
    }
    response = requests.post(NVLM_INVOKE_URL, headers=headers, json=payload, stream=stream)
    if stream:
        result_text = ""
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode("utf-8")
                print(decoded_line)
                result_text += decoded_line
    else:
        result = response.json()
        print("NVIDIA VLM response:", json.dumps(result, indent=2))
        result_text = result.get("result", "")

    delete_asset(asset_id)
    return result_text

# ================= YOLO Detection =================
def get_yolo_detection_summary(video_file, frame_interval=30):
    """
    Processes the video with a YOLO detector trained for autonomous vehicles.
    Extracts frames at a given interval and returns a summary string of detected objects.
    """
    model = YOLO("yolo_autonomous.pt")
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error opening video file:", video_file)
        return ""

    frame_count = 0
    detections = {}
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            results = model(frame)
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls.item())
                    class_name = result.names.get(cls_id, str(cls_id)) if hasattr(result, "names") else str(cls_id)
                    detections[class_name] = detections.get(class_name, 0) + 1
        frame_count += 1
    cap.release()

    if not detections:
        summary = "No objects detected by YOLO."
    else:
        parts = [f"{count} instance(s) of {obj}" for obj, count in detections.items()]
        summary = "Detected objects: " + ", ".join(parts) + "."

    print("YOLO detection summary:", summary)
    return summary

# ================= Query ChatGPT (OpenAI) =================
def query_chatgpt(prompt):
    """
    Sends the combined scene description, YOLO output, and question to ChatGPT.
    The prompt instructs the model to show its reasoning and then output the final answer
    enclosed within <ans> and </ans> tags.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Alternatively, use "gpt-3.5-turbo"
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=512,
    )
    answer = response["choices"][0]["message"]["content"]
    return answer

# ================= Main Processing Function =================
def process_dataset():
    # Set paths for the CSV file and video directory (adjust as needed)
    QUESTIONS_CSV = "questions.csv"
    VIDEO_DIR = "./videos"

    results = []

    # Open and read the CSV file containing 'id' and 'question' columns.
    with open(QUESTIONS_CSV, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            vid_id = row['id']
            question = row['question']
            video_path = os.path.join(VIDEO_DIR, f"{vid_id}.mp4")

            if not os.path.exists(video_path):
                print(f"Video file {video_path} not found. Skipping id {vid_id}.")
                continue

            print(f"\nProcessing video for id: {vid_id}")
            print(f"Question: {question}")

            # Step 1: Get scene description using NVIDIA VLM.
            scene_description = get_scene_description(video_path, query="Describe the scene")
            print("Scene Description:")
            print(scene_description)

            # Step 2: Get YOLO detection summary.
            yolo_summary = get_yolo_detection_summary(video_path)
            print("YOLO Summary:")
            print(yolo_summary)

            # Combine the information with the question.
            combined_text = (
                f"Scene Description: {scene_description}\n"
                f"YOLO Summary: {yolo_summary}\n"
            )
            # Instruct ChatGPT to reason out its answer and provide the final one-letter answer within <ans> tags.
            prompt = (
                f"{combined_text}\n"
                f"Based on the above, please reason through your thinking and then output the final answer "
                f"as one of the following letters (A, B, C, or D) enclosed in <ans> and </ans> tags.\n"
                f"Question: {question}\n"
            )

            # Step 3: Query ChatGPT.
            chatgpt_response = query_chatgpt(prompt)
            print("ChatGPT Response:")
            print(chatgpt_response)

            # Use regex to extract the answer enclosed in <ans> ... </ans>.
            match = re.search(r"<ans>\s*([A-Da-d])\s*</ans>", chatgpt_response)
            if match:
                answer_letter = match.group(1).upper()
                print(f"Extracted Answer for id {vid_id}: {answer_letter}")
            else:
                print(f"Could not extract answer for id {vid_id}.")
                answer_letter = ""

            # Append the result while preserving the original id.
            results.append({
                'id': vid_id,
                'answer': answer_letter
            })

    # Write the results to an output CSV file.
    output_csv = "results.csv"
    with open(output_csv, "w", newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["id", "answer"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nAll results have been written to {output_csv}.")

if __name__ == "__main__":
    process_dataset()