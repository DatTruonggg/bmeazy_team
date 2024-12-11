import os
from typing import Optional
import torch
import json
from tqdm import tqdm
from pyannote.audio import Pipeline


class AudioDetection:
    def __init__(self, model_name: str, auth_token: str, device: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        self.pipeline = Pipeline.from_pretrained(model_name, use_auth_token=auth_token)
        self.pipeline = self.pipeline.to(self.device)
        
    def detect_audio(self, audio_file: str) -> list:
        """Detect speech segments in an audio file."""
        output = self.pipeline(audio_file)
        result = []
        for speech in output.get_timeline().support():
            result.append([speech.start, speech.end])
        return result

    def process_audio_files(self, audio_paths: dict, save_dir: str):
        """Process all audio files and save the detected speech segments."""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        for audio_id, audio_path in tqdm(audio_paths.items()):
            result = self.detect_audio(audio_path)
            self.save_detection_result(result, save_dir, audio_id)

    def save_detection_result(self, result: list, save_dir: str, audio_id: str):
        """Save the detected speech segments to a JSON file."""
        with open(f'{save_dir}/{audio_id}.json', 'w') as f:
            json.dump(result, f)


def get_audio_paths_from_directory(base_dir: str) -> dict:
    """Parse all audio file paths from a directory structure."""
    all_audio_paths = dict()
    for part in sorted(os.listdir(base_dir)):
        part_path = os.path.join(base_dir, part)
        audio_paths = sorted(os.listdir(part_path))
        audio_paths_dict = {
            audio_path.replace('.wav', ''): os.path.join(part_path, audio_path) 
            for audio_path in audio_paths
        }
        all_audio_paths[part] = audio_paths_dict
    return all_audio_paths


def main():
    audios_dir = './Audios'
    save_dir_all = './audio_detection'
    
    # Step 1: Parse audio file paths from the directory
    all_audio_paths = get_audio_paths_from_directory(audios_dir)
    
    # Step 2: Initialize the AudioProcessor with the pre-trained pipeline
    processor = AudioDetection(
        model_name="pyannote/voice-activity-detection",
        auth_token="your_token"
    )
    
    # Step 3: Process all audio files and save the results
    for key, audio_paths_dict in all_audio_paths.items():
        save_dir = os.path.join(save_dir_all, key)
        processor.process_audio_files(audio_paths_dict, save_dir)


if __name__ == "__main__":
    main()
