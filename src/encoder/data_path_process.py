import os
import glob

def process_data_path(keyframes_dir: str):
    all_keyframe_paths = {}
    for part in sorted(os.listdir(keyframes_dir)):
        if not part.startswith("L"):  
            continue
    
        data_part = part
        all_keyframe_paths[data_part] = {}
        data_part_path = os.path.join(keyframes_dir, data_part)

        if not os.path.isdir(data_part_path):  
            continue
        video_dirs = sorted(os.listdir(data_part_path))
        for video_dir in video_dirs:
            if not video_dir.startswith(f"{data_part}_"):  
                continue
            video_id = video_dir.split('_')[-1]
            video_dir_path = os.path.join(data_part_path, video_dir)
            keyframe_paths = sorted(glob.glob(os.path.join(video_dir_path, '*.jpg')))
            all_keyframe_paths[data_part][video_id] = keyframe_paths

    return all_keyframe_paths
