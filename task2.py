#sudo apt update
#sudo apt install ffmpeg
#pip install pydub

import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from pydub import AudioSegment

# Your input and output file paths
input_file = "brrr.m4a"         # Replace with your file name
output_file = "output.wav"       

# Load the M4A file
audio = AudioSegment.from_file(input_file, format="m4a")

# Optional: Set to mono and 16kHz for models like wav2vec2
audio = audio.set_channels(1).set_frame_rate(16000)

# Export to WAV format
audio.export(output_file, format="wav")

# Load pretrained model and tokenizer
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Load audio
filename = "output.wav"  # Must be 16kHz mono .wav file
waveform, sample_rate = torchaudio.load(filename)


# Tokenize
input_values = tokenizer(waveform.squeeze().numpy(), return_tensors="pt").input_values

# Run inference
with torch.no_grad():
    logits = model(input_values).logits

# Decode
predicted_ids = torch.argmax(logits, dim=-1)
transcription = tokenizer.decode(predicted_ids[0])

print("Transcription:", transcription)
