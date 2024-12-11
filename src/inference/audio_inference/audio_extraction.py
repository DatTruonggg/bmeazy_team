import os
import subprocess
from tqdm import tqdm
from typing import Dict


class AudioExtraction:
    def __init__(self, videos_dir: str, output_dir: str, output_ext: str = "wav"):
        """
        Initializes the converter with the input video directory and output directory.

        :param videos_dir: Path to the directory containing video files.
        :param output_dir: Path to the directory where audio files will be saved.
        :param output_ext: File extension for the output audio files (default: "wav").
        """
        self.videos_dir = videos_dir
        self.output_dir = output_dir
        self.output_ext = output_ext
        self.all_video_paths: Dict[str, Dict[str, str]] = {}

    def parse_video_info(self) -> None:
        """
        Parses the directory structure of videos and organizes them into a dictionary.
        """
        for part in sorted(os.listdir(self.videos_dir)):
            data_part = part.split('_')[-1]  # Extract L01, L02, etc.
            self.all_video_paths[data_part] = {}

            data_part_path = os.path.join(self.videos_dir, part, "video")
            video_paths = sorted(os.listdir(data_part_path))
            video_ids = [video_path.replace('.mp4', '').split('_')[-1] for video_path in video_paths]

            for video_id, video_path in zip(video_ids, video_paths):
                video_path_full = os.path.join(data_part_path, video_path)
                self.all_video_paths[data_part][video_id] = video_path_full

    @staticmethod
    def convert_video_to_audio_ffmpeg(video_file: str, save_path: str, output_ext: str = "wav") -> None:
        """
        Converts a video file to audio using ffmpeg.

        :param video_file: Path to the video file.
        :param save_path: Path to save the converted audio file (without extension).
        :param output_ext: File extension for the output audio file.
        """
        subprocess.call(
            ["ffmpeg", "-y", "-i", video_file, f"{save_path}.{output_ext}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT
        )

    def convert_all_videos_to_audio(self) -> None:
        """
        Converts all videos in the parsed video structure to audio files.
        """
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        for key in tqdm(self.all_video_paths.keys(), desc="Processing Parts"):
            save_dir = os.path.join(self.output_dir, key)

            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            video_paths_dict = self.all_video_paths[key]
            for video_id, video_path in tqdm(video_paths_dict.items(), desc=f"Converting {key}", leave=False):
                save_path = os.path.join(save_dir, video_id)
                self.convert_video_to_audio_ffmpeg(video_path, save_path, self.output_ext)

    def run(self) -> None:
        """
        Runs the complete video-to-audio conversion process.
        """
        self.parse_video_info()
        self.convert_all_videos_to_audio()


# Example usage
if __name__ == "__main__":
    videos_dir = './data/video'
    save_dir_all = './data/Audios'
    converter = AudioExtraction(videos_dir, save_dir_all, output_ext="wav")
    converter.run()
