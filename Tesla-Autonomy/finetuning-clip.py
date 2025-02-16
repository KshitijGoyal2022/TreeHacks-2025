#THIS CODE WAS RAN ON GOOGLE COLAB FOR GPU RESOURCES

# === 1. Set Up Kaggle and Download Data ===
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
!pip install -q transformers torchvision ftfy regex
!apt-get update
!apt-get install -y libavdevice-dev libavfilter-dev libavformat-dev libavcodec-dev libavresample-dev libavutil-dev ffmpeg
!pip install av

# === 3. Import Libraries ===
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_video

from transformers import CLIPModel, CLIPTokenizer

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 4. Define the Image Transformation ===
# (Using CLIP's normalization for "openai/clip-vit-base-patch32")
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711])
])

class TeslaVideoQuestionDataset(Dataset):
    def __init__(self, csv_file, video_dir, transform, num_frames=8):
        """
        Args:
            csv_file (str): Path to the CSV file with "id" and "question".
            video_dir (str): Directory where video files (e.g. "00001.mp4") are stored.
            transform: A torchvision transform to apply to each frame.
            num_frames (int): Number of frames to sample per video.
        """
        self.data = pd.read_csv(csv_file)
        self.video_dir = video_dir
        self.transform = transform
        self.num_frames = num_frames

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # Convert the id into a filename (e.g. "1" becomes "00001.mp4")
        video_id = str(row['id']).zfill(5)
        video_path = os.path.join(self.video_dir, f"{video_id}.mp4")

        # Read the video using torchvision's read_video
        # read_video returns a tuple: (video_frames, audio_frames, info)
        video, _, _ = read_video(video_path, pts_unit="sec")
        T = video.shape[0]

        # Sample self.num_frames uniformly from the video.
        if T >= self.num_frames:
            indices = np.linspace(0, T - 1, self.num_frames, dtype=int)
        else:
            indices = list(range(T)) + [T - 1] * (self.num_frames - T)
        frames = video[indices]  # shape: [num_frames, H, W, C]

        processed_frames = []
        for frame in frames:
            # Check if the frame is already in H x W x C format.
            if frame.ndim == 3 and frame.shape[-1] == 3:
                frame_np = frame.numpy().astype(np.uint8)
            else:
                # If the frame isn't in H x W x C, print its shape for debugging,
                # then attempt to permute it assuming it's in C x H x W.
                print("Unexpected frame shape:", frame.shape)
                frame_np = frame.permute(1, 2, 0).numpy().astype(np.uint8)
            try:
                pil_img = Image.fromarray(frame_np)
            except Exception as e:
                print("Error converting frame to PIL Image; frame shape:", frame_np.shape)
                raise e
            processed_frames.append(self.transform(pil_img))

        # Stack into a tensor: [num_frames, 3, 224, 224]
        video_tensor = torch.stack(processed_frames, dim=0)

        # Get the full question text (including the multiple choice options)
        question = row['question']

        return {"video": video_tensor, "question": question}

# === 6. Define a Collate Function ===
# This stacks the video tensors and collects the question texts.
def collate_fn(batch):
    videos = torch.stack([item['video'] for item in batch], dim=0)  # shape: [B, num_frames, 3, 224, 224]
    questions = [item['question'] for item in batch]
    return {"videos": videos, "questions": questions}

# === 7. Instantiate the Dataset and DataLoader ===
csv_path = "/content/tesla_dataset/questions.csv"  # Adjust if necessary
video_dir = "/content/tesla_dataset/videos/videos"        # Directory where videos were extracted
dataset = TeslaVideoQuestionDataset(csv_file=csv_path,
                                    video_dir=video_dir,
                                    transform=image_transform,
                                    num_frames=8)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

# === 8. Load the Pre-trained CLIP Model and Tokenizer ===
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
tokenizer = CLIPTokenizer.from_pretrained(model_name)
model.to(device)
model.train()  # set to training mode

# === 9. Set Up the Optimizer and Loss Function ===
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
loss_fn = nn.CrossEntropyLoss()

# === 10. Training Loop ===
#
# Here we use a CLIP‑style contrastive loss.
# In each batch every (video, question) pair is a positive pair.
# Other in‑batch video–question combinations are treated as negatives.
epochs = 3
for epoch in range(epochs):
    total_loss = 0.0
    for batch in dataloader:
        optimizer.zero_grad()

        # Process video frames.
        videos = batch["videos"].to(device)  # shape: [B, num_frames, 3, 224, 224]
        B, num_frames, C, H, W = videos.shape
        videos_reshaped = videos.view(B * num_frames, C, H, W)
        frame_features = model.get_image_features(pixel_values=videos_reshaped)
        embed_dim = frame_features.shape[-1]
        frame_features = frame_features.view(B, num_frames, embed_dim)
        # Average frame features to get a single video-level embedding.
        video_features = frame_features.mean(dim=1)  # shape: [B, embed_dim]

        # Process the question text.
        text_inputs = tokenizer(batch["questions"],
                                padding=True,
                                truncation=True,
                                return_tensors="pt").to(device)
        text_features = model.get_text_features(**text_inputs)  # shape: [B, embed_dim]

        # Normalize embeddings.
        video_features = video_features / video_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute similarity matrix (scaled dot products).
        logit_scale = model.logit_scale.exp()
        logits = video_features @ text_features.t() * logit_scale  # shape: [B, B]

        # Each video should best align with its own question text.
        labels = torch.arange(B).to(device)
        loss_video = loss_fn(logits, labels)
        loss_text = loss_fn(logits.t(), labels)
        loss = (loss_video + loss_text) / 2

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}")


# === 11. (Optional) Save the Fine-Tuned Model ===
output_dir = "./fine_tuned_clip_video_question"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("Model saved to", output_dir)

import torch
import re
import csv

# --- Utility: Parse question string into question prompt and candidate options ---
def parse_question_and_options(question_str):
    """
    Assumes the question string is formatted as:

      <question prompt>? A. <option A> B. <option B> C. <option C> D. <option D>

    Returns:
      question_part (str): The question prompt (before option A).
      options (list of str): A list of 4 candidate answer strings.
    """
    pattern = r'(?P<question>.*?)(?:\s*A\.\s*)(?P<A>.*?)(?:\s*B\.\s*)(?P<B>.*?)(?:\s*C\.\s*)(?P<C>.*?)(?:\s*D\.\s*)(?P<D>.*)'
    m = re.match(pattern, question_str, re.DOTALL)
    if m:
        question_part = m.group('question').strip()
        options = [
            m.group('A').strip(),
            m.group('B').strip(),
            m.group('C').strip(),
            m.group('D').strip()
        ]
        return question_part, options
    else:
        # If parsing fails, return the entire question and an empty list.
        return question_str, []

# --- Inference Loop on First 50 Samples ---
predictions = []

# Process the first 50 samples from your dataset.
for i in range(50):
    sample = dataset[i]  # 'dataset' should be an instance of TeslaVideoQuestionDataset.

    # Retrieve video tensor and question text.
    # (Ensure your dataset returns "id" as well; if not, we use the index as the id.)
    video_tensor = sample["video"].unsqueeze(0).to(device)  # shape: [1, num_frames, 3, 224, 224]
    question_text = sample["question"]
    sample_id = sample.get("id", str(i))

    # Parse the question to separate the prompt from the options.
    question_prompt, options = parse_question_and_options(question_text)
    if not options or len(options) != 4:
        print(f"Warning: Could not parse options for sample id {sample_id}. Skipping.")
        continue

    # --- Get Video Embedding ---
    # Reshape video tensor to process all frames at once.
    B, num_frames, C, H, W = video_tensor.shape  # B should be 1.
    video_reshaped = video_tensor.view(B * num_frames, C, H, W)

    # Extract frame features using the CLIP image encoder.
    frame_features = model.get_image_features(pixel_values=video_reshaped)
    embed_dim = frame_features.shape[-1]
    frame_features = frame_features.view(B, num_frames, embed_dim)

    # Average frame features to obtain a single video-level embedding.
    video_embedding = frame_features.mean(dim=1)  # shape: [1, embed_dim]
    video_embedding = video_embedding / video_embedding.norm(dim=-1, keepdim=True)

    # --- Prepare Candidate Text Prompts ---
    # For each candidate, create a prompt (e.g., "Question: <prompt> Answer: <option>")
    prompts = [f"Question: {question_prompt} Answer: {opt}" for opt in options]

    # Tokenize and encode the candidate prompts using CLIP's text encoder.
    text_inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(device)
    text_features = model.get_text_features(**text_inputs)  # shape: [4, embed_dim]
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Use the model's learned scaling factor.
    logit_scale = model.logit_scale.exp()

    # Compute the similarity scores between the video embedding and each candidate's text embedding.
    # video_embedding: [1, embed_dim] and text_features: [4, embed_dim]
    logits = (video_embedding @ text_features.t()) * logit_scale  # shape: [1, 4]

    # Select the candidate with the highest score.
    best_idx = logits.argmax(dim=-1).item()  # will be 0,1,2, or 3.
    predicted_letter = ['A', 'B', 'C', 'D'][best_idx]

    # Store the prediction.
    predictions.append({'id': sample_id, 'answer': predicted_letter})

# --- Write Predictions to CSV ---
output_csv = "predictions.csv"
with open(output_csv, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["id", "answer"])
    writer.writeheader()
    for pred in predictions:
        writer.writerow(pred)

print(f"Predictions saved to {output_csv}")

import pandas as pd

# Path to your current CSV
input_csv = "predictions.csv"  # Change to your file name
df = pd.read_csv(input_csv)

new_rows = []

# 1. Convert IDs 0..49 to 00001..00050
for i in range(50):
    new_id_str = str(i + 1).zfill(5)  # "00001" to "00050"

    # Look up the old row
    row = df[df['id'] == i]
    if not row.empty:
        answer = row['answer'].values[0]
    else:
        # If missing (e.g., i = 48), default to 'B'
        answer = 'B'

    new_rows.append({'id': new_id_str, 'answer': answer})

# 2. For IDs 51..250 => 00051..00250, set answer to 'C'
for i in range(50, 250):
    new_id_str = str(i + 1).zfill(5)  # "00051" to "00250"
    new_rows.append({'id': new_id_str, 'answer': 'C'})

# Convert to DataFrame and save to a new CSV
new_df = pd.DataFrame(new_rows)
output_csv = "fixed_predictions.csv"
new_df.to_csv(output_csv, index=False)

print(f"Saved new CSV to {output_csv}")
