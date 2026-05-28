import os
import torch
import numpy as np
import librosa
from transformers import HubertModel, Wav2Vec2FeatureExtractor
from src.utils.util import resample_audio

class AudioProcessor:
    def __init__(self, cfg_path=None, is_training=False):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # FIX: Вычисляем абсолютный путь от корня проекта, независимо от CWD
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
        model_path = os.path.join(base_dir, 'pretrain_weights', 'audio', 'chinese-hubert-base')
        
        self.audio_encoder, self.feature_extractor = self.load_model(
            model_weight_path=model_path,
            device=self.device
        )
        print(f"AudioProcessor initialized on {self.device} | Path: {model_path}")

    def load_model(self, model_weight_path, device='cpu'):
        print(f"Loading HuBERT model from {model_weight_path}...")
        # local_files_only=False разрешает скачать с HF, если локально нет
        kwargs = {"local_files_only": False}
        
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_weight_path, **kwargs)
        audio_encoder = HubertModel.from_pretrained(model_weight_path, **kwargs).to(device=device)
        audio_encoder.eval()
        
        print("✅ HuBERT model loaded successfully.")
        return audio_encoder, feature_extractor

    def get_long_audio_emb(self, audio_path, target_sr=16000):
        waveform, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        if sr != target_sr:
            waveform = resample_audio(waveform, orig_sr=sr, target_sr=target_sr)
            
        inputs = self.feature_extractor(
            waveform, sampling_rate=target_sr, return_tensors="pt", padding=True
        ).to(self.device)
        
        with torch.no_grad():
            embeddings = self.audio_encoder(inputs.input_values).last_hidden_state
        return embeddings.squeeze(0)

    def add_silent_audio(self, audio_path, silent_audio_path=None, add_duration=2, linear_fusion=False, mode="post"):
        waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
        add_frames = int(add_duration * sr)
        silent_wave = np.zeros(add_frames, dtype=np.float32)
        
        if mode == "post":
            padded = np.concatenate([waveform, silent_wave])
        elif mode == "pre":
            padded = np.concatenate([silent_wave, waveform])
        else:
            padded = np.concatenate([silent_wave, waveform, silent_wave])
            
        import soundfile as sf
        temp_path = audio_path.replace('.wav', '_padded.wav').replace('.mp3', '_padded.wav')
        sf.write(temp_path, padded, 16000)
        return temp_path, add_duration
