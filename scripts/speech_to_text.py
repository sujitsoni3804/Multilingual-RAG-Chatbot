import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, WhisperProcessor, WhisperForConditionalGeneration
import tempfile
import os
from pathlib import Path

MODELS = {
    'gujarati': ('indicwav2vec_v1_gujarati', 'wav2vec'),
    'hindi': ('indicwav2vec_hindi', 'wav2vec'),
    'english': ('whisper_large_v3_turbo', 'whisper')
}
MODELS_BASE = Path(__file__).resolve().parent.parent / "Models"

class SpeechConverter:
    def __init__(self):
        self.models = {}
        self.processors = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loaded = False
    
    def load_models(self):
        if self.loaded:
            return
        for lang, (model_name, model_type) in MODELS.items():
            model_path = MODELS_BASE / model_name
            if not model_path.exists():
                raise FileNotFoundError(f"Local model folder not found: {model_path}")
            if model_type == 'whisper':
                self.processors[lang] = WhisperProcessor.from_pretrained(str(model_path))
                self.models[lang] = WhisperForConditionalGeneration.from_pretrained(str(model_path))
            else:
                self.processors[lang] = Wav2Vec2Processor.from_pretrained(str(model_path))
                self.models[lang] = Wav2Vec2ForCTC.from_pretrained(str(model_path))
            self.models[lang].to(self.device)
        self.loaded = True
    
    def transcribe(self, audio_file, language='gujarati'):
        if not self.loaded:
            self.load_models()
        
        if language not in self.models:
            raise ValueError('Invalid language')
        
        model = self.models[language]
        processor = self.processors[language]
        model_type = MODELS[language][1]
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            audio_file.save(tmp_file.name)
            tmp_path = tmp_file.name
        
        try:
            waveform, sample_rate = torchaudio.load(tmp_path)
            
            if waveform.shape[1] < 400:
                raise ValueError('Audio too short')
            
            if sample_rate != 16000:
                waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
            
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            if waveform.shape[1] < 1600:
                waveform = torch.nn.functional.pad(waveform, (0, 1600 - waveform.shape[1]))
            
            if model_type == 'whisper':
                input_features = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_features.to(self.device)
                with torch.no_grad():
                    predicted_ids = model.generate(input_features)
                transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            else:
                input_values = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt", padding=True).input_values.to(self.device)
                with torch.no_grad():
                    logits = model(input_values).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = processor.batch_decode(predicted_ids)[0]
            
            return transcription
        
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

speech_converter = SpeechConverter()