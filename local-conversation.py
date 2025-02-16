import time
import wave
import os
import numpy as np
import soundfile as sf
import pyaudio
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
import pyttsx3  # For text-to-speech

# Load environment variables from .env file
load_dotenv()

# OpenAI Configuration
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Audio Configuration
CHUNK = 8192        # Larger chunk size for stability
RATE = 44100       # Standard sample rate
CHANNELS = 1       # Mono audio
BUFFER_SIZE = 20   # For receiving

# Initialize PyAudio for audio recording
p = pyaudio.PyAudio()

# Audio input/output configuration
def record_audio():
    """Record audio from the microphone and save to a WAV file."""
    print("Recording your voice...")
    stream = p.open(format=pyaudio.paInt16,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    frames = []
    try:
        while True:
            data = stream.read(CHUNK)
            frames.append(data)
            if len(frames) > 20:  # Stop recording after a short period
                break
    except Exception as e:
        print(f"Error recording audio: {e}")

    stream.stop_stream()
    stream.close()

    # Save the recording to a file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"recorded_audio_{timestamp}.wav"
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    print(f"Saved audio to {filename}")
    return filename

def process_audio_and_get_response(audio_file):
    """Process audio through Whisper and GPT, return response as audio."""
    print("Converting audio to text...")
    with open(audio_file, "rb") as f:
        transcription = client.audio.transcriptions.create(
            model="whisper-1", 
            file=f,
            response_format="text",
            temperature=0.0,  # More deterministic for faster processing
            language="en"
        )
    print(f"Transcribed text: {transcription['text']}")
    
    print("Getting GPT response...")
    response_text = ""
    # Stream the chat completion
    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Keep your responses concise and natural, as they will be spoken back to the user."},
            {"role": "user", "content": transcription['text']}
        ],
        stream=True,  # Enable streaming
        temperature=0.7,
        max_tokens=150  # Limit response length for faster processing
    )
    
    print("Streaming response: ", end='', flush=True)
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            response_text += chunk.choices[0].delta.content
            print(".", end='', flush=True)
    print(f"\nGPT response: {response_text}")
    
    print("Converting response to speech...")
    
    # Initialize text-to-speech engine (Pyttsx3)
    engine = pyttsx3.init()
    engine.save_to_file(response_text, "response_audio.wav")
    engine.runAndWait()

    print(f"Response audio saved to response_audio.wav")
    return "response_audio.wav"

def play_audio(filename):
    """Play the audio file."""
    print(f"Playing audio: {filename}")
    os.system(f"start {filename}")  # Works on Windows; replace with `afplay` on macOS or `aplay` on Linux

def main():
    while True:
        try:
            # Record audio from the user
            audio_file = record_audio()
            
            if audio_file:
                # Process audio and get response
                response_file = process_audio_and_get_response(audio_file)
                
                # Play the response audio
                play_audio(response_file)
                
            # Ask if user wants to continue
            response = input("\nDo you want to continue? (y/n): ")
            if response.lower() != 'y':
                break
                
        except KeyboardInterrupt:
            print("\nStopped by user")
            break
        except Exception as e:
            print(f"Error: {e}")
            response = input("\nDo you want to try again? (y/n): ")
            if response.lower() != 'y':
                break

if __name__ == '__main__':
    main()
