from fastapi import FastAPI, File, UploadFile
import os
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch

# Initialize FastAPI app
app = FastAPI()

# Load Wav2Vec2 Model and Processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

# Get temporary directory from environment variable or use default 'temp'
TEMP_DIR = os.getenv("TEMP_DIR", "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

# Function to transcribe the audio file
def transcribe_audio(file_path):
    audio_input, sample_rate = sf.read(file_path)
    
    # Ensure the sample rate is 16kHz
    if sample_rate != 16000:
        raise ValueError("Audio must be sampled at 16kHz")
    
    inputs = processor(audio_input, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    
    return transcription

# Ping API to check service health
@app.get("/ping")
async def ping():
    return {"message": "pong"}

# ASR API to transcribe audio and return transcription
@app.post("/asr")
async def asr(file: UploadFile = File(...)):
    # Remove directories and keep only the file name
    filename = os.path.basename(file.filename)
    file_location = os.path.join(TEMP_DIR, filename)
    
    # Save the uploaded audio file temporarily
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())

    try:
        # Transcribe the audio
        transcription = transcribe_audio(file_location)
        
        # Get audio duration
        f = sf.SoundFile(file_location)
        duration = len(f) / f.samplerate

        return {
            "transcription": transcription,
            "duration": f"{duration:.1f}"  # Format duration to 1 decimal place
        }
    finally:
        # Delete the processed audio file after transcription
        os.remove(file_location)
