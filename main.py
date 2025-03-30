import os
import platform
import numpy as np
import torch
import sounddevice as sd
import simpleaudio as sa
from pyannote.audio import Pipeline
from transformers import pipeline

def clear_screen():
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEG_PIPE = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="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",
).to(DEVICE)
STT_PIPE = pipeline("automatic-speech-recognition", model="vinai/PhoWhisper-base", device=DEVICE)
TTS_PIPE = pipeline("text-to-speech", model="facebook/mms-tts-vie", device=DEVICE)

READ_SAMPLE_RATE = 16000
CHUNK_SIZE = 1
SILENCE_DURATION = 0.5 # CHUNKSIZE < SILENCE_DURATION

# Buffer
STT_PIPE({'array': np.array([0.1]), 'sampling_rate': 1})['text']

clear_screen()

while True:
    frames = []
    print(u'START!')
    with sd.InputStream(samplerate=READ_SAMPLE_RATE, channels=1) as stream:
        while True:
            data, _ = stream.read(int(READ_SAMPLE_RATE*CHUNK_SIZE))
            data = torch.from_numpy(data).to(DEVICE)
            data = data.reshape(1, -1)
            data /= torch.max(torch.abs(data))
            diarization = SEG_PIPE({"waveform": data, "sample_rate": READ_SAMPLE_RATE}, num_speakers=1)
            segments = list(diarization.itersegments())
            if (len(segments) > 0 and CHUNK_SIZE - segments[-1].end < SILENCE_DURATION):
                frames.append(data)
            else:
                break
    if len(frames) == 0:
        print('No voice detected!')
        break
    waveform = torch.cat(frames, dim=-1)
    input_text = STT_PIPE({'array': waveform.squeeze().cpu().numpy(), 'sampling_rate': READ_SAMPLE_RATE})['text']
    print(input_text)
    output = TTS_PIPE(input_text)
    waveform = np.array(output["audio"])
    sample_rate = output["sampling_rate"]
    waveform = (waveform * 32767).astype(np.int16)
    play_obj = sa.play_buffer(waveform, 1, 2, sample_rate)
    play_obj.wait_done()