import os
import math
import numpy as np 
from dotenv import load_dotenv
load_dotenv

import pytest
from src.utils.text_search import TextSearch


@pytest.fixture
def text_search_intance():
    json_path = os.getenv("ID2IMG") 
    json_path_cloud = os.getenv("ID2IMG_CLOUD")
    clipb16_bin = os.getenv("FAISS_CLIP_B16")
    clipv2_l14_bin = os.getenv("FAISS_CLIPV2_L14")
    clipv2_h14_bin = os.getenv("FAISS_CLIPV2_H14")
    audio_json_path = os.getenv("AUDO_ID2IMG_FPS")
    img2audio_json_path = os.getenv("IMG_ID2AUDIO_ID")
    return TextSearch(json_path, json_path_cloud, clipb16_bin, clipv2_l14_bin, clipv2_h14_bin, audio_json_path, img2audio_json_path)
    
def test_text_search(text_search_intance):
    text_query = "Cat Image"
    top_k = 5
    model_type = "clipv2_h14"
    storage = "local"
    
    scores, idx_image, infos_query, image_paths = text_search_intance.text_search(text_query, top_k, model_type, storage)

    print("Scores:", scores)
    print("Image indices:", idx_image)
    print("Image paths:", image_paths)
    print("Infos query:", infos_query)

def test_api():
    pass

def test_image_search():
    pass

def test_ocr_search():
    pass

def test_color_search():
    pass

def test_audio_search():
    pass

def test_tag_search():
    pass

def test_od_search():
    pass

