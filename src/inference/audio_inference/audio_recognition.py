import os
import json
import zipfile
import kenlm
import torch
import librosa
import soundfile as sf
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from pyctcdecode import Alphabet, BeamSearchDecoderCTC, LanguageModel

class AudioRecognition:
    def __init__(self, model_name: str, tokenizer_name: str, lm_path: str, device: str = None):
        # Set device for model (cuda if available else cpu)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        
        # Load Wav2Vec2 model and processor
        self.processor = Wav2Vec2Processor.from_pretrained(tokenizer_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)
        
        # Load n-gram Language Model for CTC decoding
        self.lm_model = kenlm.Model(lm_path)
        self.decoder = self.get_decoder_ngram_model(self.processor.tokenizer, lm_path)

    def get_decoder_ngram_model(self, tokenizer, ngram_lm_path):
        """Build a BeamSearch decoder using n-gram language model."""
        vocab_dict = tokenizer.get_vocab()
        vocab_list = sorted((value, key) for (key, value) in vocab_dict.items())[:-2]
        vocab_list = [x[1] for x in vocab_list]
        
        vocab_list[tokenizer.pad_token_id] = ""  # Blank character
        vocab_list[tokenizer.unk_token_id] = ""  # Unknown character
        vocab_list[tokenizer.word_delimiter_token_id] = " "  # Space character
        
        alphabet = Alphabet.build_alphabet(vocab_list, ctc_token_idx=tokenizer.pad_token_id)
        language_model = LanguageModel(self.lm_model)
        return BeamSearchDecoderCTC(alphabet, language_model=language_model)

    def transcribe_audio(self, audio_path: str, shot_segments: list, sampling_rate: int = 16000) -> list:
        """Transcribe audio by segmenting and processing each part."""
        speech, _ = librosa.load(audio_path, mono=True, sr=sampling_rate)
        speech_len = len(speech)
        results = []
        
        for start, end in shot_segments:
            audio_segments = self.segment_audio(speech, start, end, speech_len, sampling_rate)
            if audio_segments:
                transcribed_text = self.decode_audio(audio_segments, sampling_rate)
                results.append(transcribed_text)
            else:
                results.append("")
        
        return results

    def segment_audio(self, speech, start, end, speech_len, sampling_rate):
        """Segment audio into frames for processing."""
        segments = []
        while (end - start) >= 1:
            if (end - start) <= 10:
                segments.append(speech[int(start * sampling_rate):min(speech_len, round(end * sampling_rate))])
                break
            else:
                segments.append(speech[int(start * sampling_rate):min(speech_len, round((start + 10) * sampling_rate))])
                start += 10
        return segments

    def decode_audio(self, audio_segments, sampling_rate):
        """Decode the segmented audio using the model and language model."""
        input_values = self.processor(audio_segments, sampling_rate=sampling_rate, return_tensors="pt", padding="longest").input_values.to(self.device)
        logits = self.model(input_values).logits
        decoded_result = []
        for logit in logits:
            beam_search_output = self.decoder.decode(logit.cpu().detach().numpy(), beam_width=500)
            decoded_result.append(beam_search_output)
        return " ".join(decoded_result)

    def save_transcriptions(self, results, save_path: str, audio_id: str):
        """Save transcribed results to a JSON file."""
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False)


class AudioDatasetProcessor:
    def __init__(self, base_dir: str, save_dir: str):
        self.base_dir = base_dir
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def get_audio_paths(self) -> dict:
        """Retrieve all audio file paths from the base directory."""
        all_audio_paths = {}
        for part in sorted(os.listdir(self.base_dir)):
            part_path = os.path.join(self.base_dir, part)
            audio_files = sorted(os.listdir(part_path))
            audio_paths = {audio.replace('.wav', ''): os.path.join(part_path, audio) for audio in audio_files}
            all_audio_paths[part] = audio_paths
        return all_audio_paths

    def get_shot_segments(self, shot_file_path: str) -> list:
        """Retrieve the shot segments from the JSON file."""
        with open(shot_file_path, 'r') as f:
            return json.load(f)

    def process_audio_files(self, processor: AudioRecognition, all_audio_paths: dict):
        """Process all audio files by transcribing and saving results."""
        for key, audio_paths_dict in tqdm(all_audio_paths.items()):
            save_dir = os.path.join(self.save_dir, key)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            for audio_id, audio_path in tqdm(audio_paths_dict.items()):
                shot_file_path = f'/kaggle/input/aic-audiodetectionb3-2/{key}/{audio_id}.json'
                shot_segments = self.get_shot_segments(shot_file_path)
                transcription_results = processor.transcribe_audio(audio_path, shot_segments)
                save_path = os.path.join(save_dir, f'{audio_id}.json')
                processor.save_transcriptions(transcription_results, save_path, audio_id)


def main():
    # Initialize models and processors
    model_name = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
    tokenizer_name = model_name
    lm_path = '/kaggle/working/vi_lm_4grams.bin'
    processor = AudioRecognition(model_name, tokenizer_name, lm_path)
    
    # Download and extract the language model
    lm_file = hf_hub_download(model_name, filename='vi_lm_4grams.bin.zip')
    with zipfile.ZipFile(lm_file, 'r') as zip_ref:
        zip_ref.extractall('/kaggle/working/')
    
    # Load audio paths and process
    audios_dir = './Audios'
    save_dir_all = './audio'
    dataset_processor = AudioDatasetProcessor(audios_dir, save_dir_all)
    all_audio_paths = dataset_processor.get_audio_paths()
    dataset_processor.process_audio_files(processor, all_audio_paths)
    
    # Clean up language model file
    os.remove(lm_path)


if __name__ == "__main__":
    main()
