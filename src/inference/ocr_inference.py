import os
import glob
import json
from tqdm import tqdm
import easyocr

class KeyframeOCRProcessor:
    def __init__(self, keyframes_dir: str, save_dir: str, languages: list, batch_size: int = 16, gpu: bool = True):
        """
        A class to process keyframes using EasyOCR.

        Args:
            keyframes_dir (str): Path to the directory containing keyframes.
            save_dir (str): Path to the directory where OCR results will be saved.
            languages (list): List of languages for OCR (e.g., ['vi']).
            batch_size (int): Batch size for OCR inference.
            gpu (bool): Whether to use GPU for EasyOCR.
        """
        self.keyframes_dir = keyframes_dir
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.reader = easyocr.Reader(languages, gpu=gpu)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def gather_keyframe_paths(self) -> dict:
        """
        Collect all keyframe paths organized by data parts and video IDs.

        Returns:
            dict: A dictionary containing keyframe paths structured as {data_part: {video_id: [paths]}}.
        """
        all_keyframe_paths = {}
        for part in sorted(os.listdir(self.keyframes_dir)):
            data_part = part.split('_')[-1]  
            all_keyframe_paths[data_part] = {}

            data_part_path = os.path.join(self.keyframes_dir, part)
            video_dirs = sorted(os.listdir(data_part_path))
            video_ids = [video_dir.split('_')[-1] for video_dir in video_dirs]

            for video_id, video_dir in zip(video_ids, video_dirs):
                keyframe_paths = sorted(glob.glob(f"{data_part_path}/{video_dir}/*.jpg"))
                all_keyframe_paths[data_part][video_id] = keyframe_paths

        return all_keyframe_paths

    def process_videos(self, all_keyframe_paths: dict) -> None:
        """
        Process all videos for OCR and save results.

        Args:
            all_keyframe_paths (dict): Keyframe paths organized by data parts and video IDs.
        """
        for key in tqdm(sorted(all_keyframe_paths.keys()), desc="Processing data parts"):
            video_keyframe_paths = all_keyframe_paths[key]
            part_save_dir = os.path.join(self.save_dir, key)
            os.makedirs(part_save_dir, exist_ok=True)

            for video_id, keyframe_paths in tqdm(video_keyframe_paths.items(), desc=f"Processing videos in {key}"):
                ocr_results = self._process_keyframes(keyframe_paths)
                self._save_results(part_save_dir, video_id, ocr_results)

    def _process_keyframes(self, keyframe_paths: list) -> list:
        """
        Perform OCR on a list of keyframe paths.

        Args:
            keyframe_paths (list): List of keyframe image paths.

        Returns:
            list: List of OCR results for each keyframe.
        """
        ocr_results = []
        for i in range(0, len(keyframe_paths), self.batch_size):
            batch_paths = keyframe_paths[i:i + self.batch_size]
            results = self.reader.readtext_batched(batch_paths, batch_size=len(batch_paths))

            for result in results:
                refined_result = [
                    item for item in result if item[2] > 0.5  # Confidence threshold
                ]
                refined_result = easyocr.utils.get_paragraph(refined_result)
                detected_texts = [item[1] for item in refined_result]
                ocr_results.append(detected_texts)

        return ocr_results

    def _save_results(self, part_save_dir: str, video_id: str, ocr_results: list) -> None:
        """
        Save OCR results to a JSON file.

        Args:
            part_save_dir (str): Directory to save results for a data part.
            video_id (str): ID of the video.
            ocr_results (list): OCR results to save.
        """
        file_path = os.path.join(part_save_dir, f"{video_id}.json")
        with open(file_path, "w", encoding="utf-8") as jsonfile:
            json.dump(ocr_results, jsonfile, ensure_ascii=False)

if __name__ == "__main__":
    # Example usage
    keyframes_dir = './keyframeb3'
    save_dir = '/kaggle/working/ocr'

    processor = KeyframeOCRProcessor(
        keyframes_dir=keyframes_dir,
        save_dir=save_dir,
        languages=['vi'],  # Vietnamese
        batch_size=16,
        gpu=True
    )

    # Gather keyframe paths
    all_keyframe_paths = processor.gather_keyframe_paths()

    # Process videos
    processor.process_videos(all_keyframe_paths)
