from src.utils.text_search import TextSearch

from fastapi import FastAPI, status
from loguru import logger
from pydantic import BaseModel
import os 
from dotenv import load_dotenv
load_dotenv()


app = FastAPI(    
    author="Dat Truong",
    title="Text-to-image Retrieval",
    description="Text to Image retrieval using ViT Model (CLIP)",
    version="0.0.1",
)
json_path = os.getenv("ID2IMG") 
json_path_cloud = os.getenv("ID2IMG_CLOUD")
clipb16_bin = os.getenv("FAISS_CLIP_B16")
clipv2_l14_bin = os.getenv("FAISS_CLIPV2_L14")
clipv2_h14_bin = os.getenv("FAISS_CLIPV2_H14")
audio_json_path = os.getenv("AUDO_ID2IMG_FPS")
scene_path = os.getenv("SCENE_ID2INFO")
img2audio_json_path = os.getenv("IMG_ID2AUDIO_ID")
search = TextSearch(json_path, json_path_cloud, clipb16_bin, clipv2_l14_bin, clipv2_h14_bin, audio_json_path, img2audio_json_path)


class InputTextSearch(BaseModel):
    query: str
    top_k: int
    model_type: str
    storage: str 

class InputOCRSearch(BaseModel):
    text: str
    top_k: int 
    storage: str 
    
class InputASRSearch(BaseModel):
    asr_input: str
    top_k: int
    storage: str
    index: None 
    semantic: bool = True
    keyword: bool = True
    

@app.get("/healthcheck", status_code=status.HTTP_200_OK)
def healthcheck():
    return {"healthcheck": "Everything is in healthy mode!"}

@app.post("/text_search")
async def search_text_to_image(input_text: InputTextSearch):
    # Khởi tạo đối tượng TextSearch
    _, _, _, list_image_paths = search.text_search(
        text=input_text.query,
        top_k=input_text.top_k,
        model_type=input_text.model_type,
        storage=input_text.storage,
    )
    
    # Gọi phương thức tìm kiếm từ TextSearch và lấy kết quả
    
    # Trả về kết quả tìm kiếm (có thể là danh sách, tùy thuộc vào cách bạn muốn hiển thị kết quả)
    return {"results": list_image_paths}

async def search_image_to_image():
    pass

@app.post("/asr_search")
async def search_ast(input_asr: InputASRSearch):
    _, _, image_paths = search.asr_retrieval_helper(
        asr_input= input_asr.asr_input,
        k = input_asr.top_k,
        storage = input_asr.storage,
        index = input_asr.index,
        semantic = input_asr.semantic,
        keyword = input_asr.keyword
    )
    return {"results": image_paths}

@app.post("/ocr_search")
async def search_ocr(input_ocr: InputOCRSearch):
    _, _, image_paths = search.ocr_search(
        text = input_ocr.text,
        top_k = input_ocr.top_k,
        storage = input_ocr.storage
    )
    return {"results": image_paths}

async def search_od():
    pass

async def search_color():
    pass

async def search_tag():
    pass
