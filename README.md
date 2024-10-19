# Source code
```cmd
conda create -n bme python=3.11
conda activate bme
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```


change paths in `.env` file: 
```python
FAISS_CLIPV2_H14 = "/home/dattruong/dat/AI/Competition/BMEazy/data/clipv2_h14_cosine.bin"
FAISS_CLIPV2_L14 = "/home/dattruong/dat/AI/Competition/BMEazy/data/faiss_clipv2l14_cosine.bin"
FAISS_CLIP_B16 = "/home/dattruong/dat/AI/Competition/BMEazy/data/clip_b16_cosine.bin"

ID2IMG = "/home/dattruong/dat/AI/Competition/BMEazy/data/id2img_fps.json"
ID2IMG_CLOUD = "/home/dattruong/dat/AI/Competition/BMEazy/data/id2fps_cloud_v2.json"
IDFOLDER = "/home/dattruong/dat/AI/Competition/BMEazy/data/id_folder.json"

SCENE_ID2INFO = "/home/dattruong/dat/AI/Competition/BMEazy/data/scene_id2info.json"
AUDO_ID2IMG_FPS = "/home/dattruong/dat/AI/Competition/BMEazy/data/audio_id2img_id.json"
VIDEO_ID2IMG_ID = "/home/dattruong/dat/AI/Competition/BMEazy/data/video_id2img_id.json"
IMG_ID2AUDIO_ID = "/home/dattruong/dat/AI/Competition/BMEazy/data/img_id2audio_id.json"
VIDEO_DIVSION_TAG = "/home/dattruong/dat/AI/Competition/BMEazy/data/video_division_tag.json"
MAP_KEYFRAME = "/home/dattruong/dat/AI/Competition/BMEazy/data/map_keyframes.json"

ROOT_DB = "/media/dattruong/568836F88836D669/Users/DAT/Hackathon/HCMAI24/Data"
KEYFRAME_DB = "/media/dattruong/568836F88836D669/Users/DAT/Hackathon/HCMAI24/Data/Keyframe"
VIDEO_DB = "/media/dattruong/568836F88836D669/Users/DAT/Hackathon/HCMAI24/Video"

PROJECT_ROOT = "/home/dattruong/dat/AI/Competition/BMEazy/"
```