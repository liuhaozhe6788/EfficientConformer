import os
import numpy as np
import json
from pathlib import Path
import librosa
import glob
import torch
import torchaudio
from functions import create_model
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.silence import split_on_silence
import noisereduce as nr 
import soundfile as sf


# Select one of the official pretrained models
pretrained_model = "EfficientConformerCTCLarge"

config_file = "configs/" + pretrained_model + ".json"

# Load model Config
with open(config_file) as json_config:
  config = json.load(json_config)

# PyTorch Device
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print("Device:", device)

# Create and Load pretrained model
model = create_model(config).to(device)
model.summary()
model.eval()
model.load(os.path.join("callbacks", pretrained_model, "checkpoints_swa-equal-401-450.ckpt"))

while True:

  # Load audio file
  # Get the reference audio filepath
  message = "Reference voice: enter an audio filepath of a voice (mp3, " \
              "wav, m4a, flac, ...):\n"
  audio_file = input(message)
  audio, sr = librosa.load(audio_file)

  # # Plot audio
  # plt.title(audio_file.split("/")[-1])
  # plt.plot(audio[0])
  # plt.show()
  # print()

  # if audio.shape[0] == 2:
  #   audio = torch.unsqueeze(audio[1], 0)


  audio = nr.reduce_noise(audio, sr, prop_decrease=0.7)
  path_, name = os.path.split(audio_file)
  name, suffix = os.path.splitext(name)
  out_file = os.path.join(path_, name + "_nr.wav")  
  sf.write(out_file, audio.astype(np.float32), sr)

  sound = AudioSegment.from_mp3(out_file)
  os.remove(out_file)
  chunks = split_on_silence(sound, min_silence_len=1500, silence_thresh=-50, keep_silence=500)
  chunk_paths = []
  predictions = []
  for i, chunk in enumerate(chunks):
    chunk_path = os.path.join(path_, name + f"_chunk{i}.wav")
    chunk.export(chunk_path)
    chunk_paths.append(chunk_path)
    chunk_audio, sr = librosa.load(chunk_path)
    os.remove(chunk_path)
    chunk_audio = np.reshape(chunk_audio, (1, len(chunk_audio)))
    chunk_audio = torch.from_numpy(chunk_audio)
    # Predict sentence
    prediction = model.beam_search_decoding(chunk_audio.to(device), x_len=torch.tensor([len(chunk_audio[0])], device=device))[0]
    predictions.append(prediction)

  out_text = ' '.join(predictions)
  print("model output text:", out_text, '\n')
  for i in range(100):
    print('*', end='')
  print('\n')