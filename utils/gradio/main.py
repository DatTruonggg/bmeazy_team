import gradio as gr
import os
import shutil
import json
from dotenv import load_dotenv
import numpy as np 
import matplotlib
matplotlib.use('Agg')  
import boto3
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Update sys path to import TextSearch
from utils.parse_frontend import parse_data
from utils.text_search import TextSearch
from utils.context_encoding import VisualEncoding
from utils.combine_utils import merge_searching_results_by_addition
from utils.search_utils import group_result_by_video 
# Load environment variables
load_dotenv()

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=[""], allow_methods=[""], allow_headers=["*"])

VisualEncoder = VisualEncoding()
json_path = os.getenv("ID2IMG") 
json_path_cloud = os.getenv("ID2IMG_CLOUD")
clipb16_bin = os.getenv("FAISS_CLIP_B16")
clipv2_l14_bin = os.getenv("FAISS_CLIPV2_L14")
clipv2_h14_bin = os.getenv("FAISS_CLIPV2_H14")
audio_json_path = os.getenv("AUDO_ID2IMG_FPS")
scene_path = os.getenv("SCENE_ID2INFO")
img2audio_json_path = os.getenv("IMG_ID2AUDIO_ID")
video_division_path = os.getenv("VIDEO_DIVSION_TAG")
map_keyframes = os.getenv("MAP_KEYFRAME")
video_id2img_id = os.getenv("VIDEO_ID2IMG_ID")
root_db = os.getenv("ROOT_DB")


FPS = 25.0
s3 = boto3.client('s3', region_name='ap-southeast-1')
bucket_name = "bmeazy"

search = TextSearch(json_path, json_path_cloud, clipb16_bin, clipv2_l14_bin, clipv2_h14_bin, audio_json_path, img2audio_json_path)

DictKeyframe2Id = search.load_json_file(json_path)
DictKeyframe2IdCloud = search.load_json_file(json_path_cloud)

with open(map_keyframes, 'r') as f:
  KeyframesMapper = json.load(f)

# Function for text-to-image search and result display
def text_to_image(text: str, top_k: int, model_type: str, storage: str):
    global search  # Use the globally loaded model


    if model_type == "clip": 
        _, list_ids, _, list_image_paths = search.text_search(text=text, top_k=top_k, model_type="clip", storage=storage)

    elif model_type == "clipv2_l14": 
        _, list_ids, _, list_image_paths = search.text_search(text=text, top_k=top_k, model_type="clipv2_l14", storage=storage)
    elif model_type == "clipv2_h14": 
        _, list_ids, _, list_image_paths = search.text_search(text=text, top_k=top_k, model_type="clipv2_h14", storage=storage)

    elif model_type == "clip + clipv2_l14" :
        scores_clip, list_clip_ids, _, _ = search.text_search(text = text, top_k=top_k, model_type='clip', storage = storage)
        scores_clipv2_l14, list_clipv2_l14_ids, _, _ = search.text_search(text = text, top_k=top_k, model_type='clipv2_l14', storage = storage)
        _, list_ids = merge_searching_results_by_addition([scores_clip, scores_clipv2_l14],
                                                                  [list_clip_ids, list_clipv2_l14_ids])
        
        if storage == "cloud": 
            infos_query = list(map(DictKeyframe2IdCloud.get, list(list_ids)))
        elif storage == "local": 
            infos_query = list(map(DictKeyframe2Id.get, list(list_ids)))
        list_image_paths = [info['image_path'] for info in infos_query]
        
    elif model_type == "clipv2_l14 + clipv2_h14": 
        scores_clipv2_l14, list_clipv2_l14_ids, _, _ = search.text_search(text = text, top_k=top_k, model_type='clipv2_l14', storage = storage)
        scores_clipv2_h14, list_clipv2_h14_ids, _, _ = search.text_search(text = text, top_k=top_k, model_type='clipv2_h14', storage = storage)
        _, list_ids = merge_searching_results_by_addition([scores_clipv2_l14, scores_clipv2_h14],
                                                                  [list_clipv2_l14_ids, list_clipv2_h14_ids])
        if storage == "cloud": 
            infos_query = list(map(DictKeyframe2IdCloud.get, list(list_ids)))
        elif storage == "local": 
            infos_query = list(map(DictKeyframe2Id.get, list(list_ids)))        
        list_image_paths = [info['image_path'] for info in infos_query]
        
    elif model_type == "clip + clipv2_h14":
        scores_clip, list_clip_ids, _, _ = search.text_search(text = text, top_k=top_k, model_type='clip', storage = storage)
        scores_clipv2_h14, list_clipv2_h14_ids, _, _ = search.text_search(text = text, top_k=top_k, model_type='clipv2_h14', storage = storage)
        _, list_ids = merge_searching_results_by_addition([scores_clip, scores_clipv2_h14],
                                                                  [list_clip_ids, list_clipv2_h14_ids])
        if storage == "cloud": 
            infos_query = list(map(DictKeyframe2IdCloud.get, list(list_ids)))
        elif storage == "local": 
            infos_query = list(map(DictKeyframe2Id.get, list(list_ids)))        
        list_image_paths = [info['image_path'] for info in infos_query]

    # data = group_result_by_video(lst_scores, list_ids, list_image_paths, KeyframesMapper)


    # Create titles for each image
    titles = []
    img_paths = []
    if storage == "local":
        for img_path in list_image_paths:
            folder = img_path.split("/")[-2]
            img_path = os.path.join(root_db, img_path)
            img_paths.append(img_path)
            frame_idx = os.path.basename(img_path).split(".")[0].strip()  # Frame index
            
            title = f"{folder}, {frame_idx}"
            titles.append(title)
    elif storage == "cloud":
        for img_path in list_image_paths:
            # Construct the S3 URL for the image
            folder = img_path.split("/")[-2]
            frame_idx = img_path.split("/")[-1].split(".")[0].strip() 

            image_key = img_path.lstrip('/')  # Remove leading slashes for correct key formation
            full_img_path = f"https://{bucket_name}.s3.amazonaws.com/{image_key}"
            img_paths.append(full_img_path)

            # Create a title if needed
            title = f"{folder}, {frame_idx}"
            titles.append(title)
    return [(img, title) for img, title in zip(img_paths, titles)]


# Function to get idx_image based on selected image path
def get_idx_from_image(folder_name, image_name, storage):
    if storage == "local":
        short_folder = folder_name[:3]
        selected_image = os.path.join("Keyframe", short_folder, folder_name, image_name + '.jpg')  # Fixed path construction
        selected_image = selected_image.replace('\\', '/')
        for idx, info in DictKeyframe2Id.items():
            img_path = info['image_path']  # Get image path from value
            if img_path == selected_image:  
                return idx
        return 0
    
    elif storage == "cloud":
        short_folder = folder_name[:3]
        selected_image = os.path.join("keyframes", short_folder, folder_name, image_name + '.jpg')  # Fixed path construction
        selected_image = selected_image.replace('\\', '/')
        for idx, info in DictKeyframe2IdCloud.items():
            img_path = info['image_path']  # Get image path from value
            if img_path == selected_image:  
                return idx
        return 0
    
# Function to search neighboring images
def search_neighbor(folder_name, image_name, storage:str):
    if storage == "cloud":
        idx_image = get_idx_from_image(folder_name = folder_name, image_name = image_name, storage = "cloud")
    elif storage == "local":
        idx_image = get_idx_from_image(folder_name = folder_name, image_name = image_name, storage = "local")
    images = []
    
    # Find start and end indices
    start_idx = max(0, idx_image - 30)  # Do not go below 0
    end_idx = min(idx_image + 269, len(DictKeyframe2Id) - 1)  # Ensure we don't go out of bounds

    # Iterate through indices from start_idx to end_idx
    if storage == "local":
        for idx in range(start_idx, end_idx + 1):
            if idx in DictKeyframe2Id.keys():  # Check if idx is valid in string format
                image_path = DictKeyframe2Id[idx]['image_path']  # Access using string keys
                image_path = os.path.join(root_db, image_path)

                folder = os.path.basename(os.path.dirname(image_path))  # Extract folder name from path
                frame_idx = os.path.basename(image_path).split(".")[0]  # Frame index
                
                title = f"{folder}, {frame_idx}"
                images.append((image_path, title))      

    elif storage == "cloud":
        for idx in range(start_idx, end_idx + 1):
            if idx in DictKeyframe2Id.keys():  # Check if idx is valid in string format
                image_path = DictKeyframe2IdCloud[idx]['image_path']  # Access using string keys
                image_key = image_path.lstrip("/")
                full_img_path = f"https://{bucket_name}.s3.amazonaws.com/{image_key}"

                folder = os.path.basename(os.path.dirname(image_path))  # Extract folder name from path
                frame_idx = os.path.basename(image_path).split(".")[0]  # Frame index
                
                title = f"{folder}, {frame_idx}"
                images.append((full_img_path, title))    

    return images

def image_search_to_image(folder_name, image_name, top_k, storage): 
    global search
    img_paths = []
    titles = []

    # Get the query index based on the storage type
    if storage == "cloud":
        id_query = get_idx_from_image(folder_name, image_name, "cloud")
        _, _, _, image_paths = search.image_search(id_query, top_k, 'cloud')

    else:
        id_query = get_idx_from_image(folder_name, image_name, "local")
        _, _, _, image_paths = search.image_search(id_query, top_k, 'local')

    for img_path in image_paths:
        folder = img_path.split("/")[-2]
        frame_idx = os.path.basename(img_path).split(".")[0].strip()

        # Handle paths based on storage type
        if storage == "local":
            # Construct local paths
            full_img_path = os.path.join(root_db, img_path)
        elif storage == "cloud":
            # Construct S3 URL for cloud storage
            image_key = img_path.lstrip('/')  # Remove leading slashes for correct key formation
            full_img_path = f"https://{bucket_name}.s3.amazonaws.com/{image_key}"

        # Append the image path and title
        img_paths.append(full_img_path)
        title = f"{folder}, {frame_idx}"
        titles.append(title)

    return [(img, title) for img, title in zip(img_paths, titles)]


def convert_to_milisecond(frame_idx):
    frame_idx = frame_idx.lstrip("0")
    time_ms = (int(frame_idx) / FPS) *1000
    return int(time_ms)

def frame_idx_to_milliseconds(evt: gr.SelectData):  
    caption = evt.value['caption']
    if caption:
        # Tách folder và image name từ caption
        parts = caption.split(', ')
        #folder_name = parts[0]  # Lấy tên folder
        image_name = parts[1]   # Lấy tên hình ảnh
        frame_idx = image_name
        if frame_idx in ["000", "0000", "00000"]:
            frame_idx_display = "0000"  # Giữ nguyên frame_idx
        else: 
            frame_idx_display = frame_idx.lstrip("0")  # Loại bỏ các số không ở đầu
        time_ms = (int(frame_idx_display) / FPS) * 1000
    return int(time_ms)

def submit_kis_and_vis(session_id, evaluation_id, video_id, milisec_name):
    url = f"https://eventretrieval.one/api/v2/submit/{evaluation_id}"
    headers = {
        "Content-Type": "application/json"
    }
    params = {"session": session_id}
    start_time = milisec_name
    # Create the body for the POST request
    body = {
        "answerSets": [
            {
                "answers": [
                    {
                        "mediaItemName": video_id,
                        "start": start_time,
                        "end": start_time
                    }
                ]
            }
        ]
    }

    # Send POST request
    response = requests.post(url, headers=headers, params=params, json=body)

    if response.status_code == 200:
        return f"Successfully submitted KIS for video {video_id} from {start_time} to {start_time}."
    else:
        return f"Failed to submit KIS. Status code: {response.status_code}, Message: {response.text}"


def submit_qa(session_id, evaluation_id, video_id, milisec_name, answer):
    url = f"https://eventretrieval.one/api/v2/submit/{evaluation_id}"
    headers = {
        "Content-Type": "application/json"
    }
    params = {"session": session_id}
    time_ms = milisec_name
    # Format the answer according to the expected structure
    answer_qa = f"{answer}-{video_id}-{time_ms}"

    # Create the body for the POST request
    body = {
        "answerSets": [
            {
                "answers": [
                    {
                        "text": answer_qa
                    }
                ]
            }
        ]
    }

    # Send POST request
    response = requests.post(url, headers=headers, params=params, json=body)

    if response.status_code == 200:
        return f"Successfully submitted QA: {answer_qa}"
    else:
        return f"Failed to submit QA. Status code: {response.status_code}, Message: {response.text}"

def get_folder_select_index(evt: gr.SelectData):
    # Lấy caption từ sự kiện chọn
    caption = evt.value['caption']
    if caption:
        # Tách folder và image name từ caption
        parts = caption.split(', ')
        folder_name = parts[0]  # Lấy tên folder
        #image_name = parts[1]   # Lấy tên hình ảnh
    return folder_name

def get_frame_select_index(evt: gr.SelectData):
    # Lấy caption từ sự kiện chọn

    caption = evt.value['caption']
    if caption:
        # Tách folder và image name từ caption
        parts = caption.split(', ')
        #folder_name = parts[0]  # Lấy tên folder
        image_name = parts[1]   # Lấy tên hình ảnh
    return image_name

def show_icons():
    icon_paths = []
    try:
        icon_paths = [os.path.join("./assets/icons", f) for f in os.listdir("./assets/icons") if f.endswith(('.png', '.jpg', '.jpeg'))]
        if not icon_paths:
            print("No icons found in the specified directory.")
    except Exception as e:
        print(f"Error loading icons: {e}")  # In ra lỗi để kiểm tra
    return icon_paths

def show_colors():
    color_paths = []
    try:
        color_paths = [os.path.join("./assets/colors", f) for f in os.listdir("./assets/colors") if f.endswith(('.png', '.jpg', '.jpeg'))]
        if not color_paths:
            print("No colors found in the specified directory.")
    except Exception as e:
        print(f"Error loading colors: {e}")  # In ra lỗi để kiểm tra
    return color_paths



# Gradio Interface with custom CSS for the search box and button
with gr.Blocks(fill_width = True, css="""
    .input-button-container {
        display: flex;
        align-items: center;    
    }
    .input-button-container input[type="text"] {
        flex-grow: 1;
    }
    .input-button-container button {
        width: 20%; /* Set button width to 1/5th of the input box */
    }
    .icon.img img, .color.img img {
        max-height: 20px; /* Set maximum height for icons and colors */
        max-width: 20px;  /* Set maximum width for icons and colors */
        object-fit: fit; /* Ensures that the aspect ratio is maintained */
        padding: 0px;
        align-items: center;
        width: 100%; /* Đảm bảo hình ảnh chiếm toàn bộ chiều rộng của thẻ cha */
        height: auto; 
    }
""") as demo:
    
    with gr.Row():
        # Left side (3 parts): Text-to-Image search
        with gr.Column(scale = 6):
            with gr.Row():
                gr.Markdown("# BMEazy")
                session_id = gr.Textbox(show_label=False, placeholder="SessionID", scale=1)
                evaluation_id = gr.Textbox(show_label=False, placeholder="EvaluationID", scale=1)


            # First row: Text Input with Search Button inside
            with gr.Row(elem_classes="input-button-container"):
                text_input = gr.Textbox(show_label=False, placeholder="Enter your query...", scale=8)
                search_button = gr.Button("Search", scale=2)
            with gr.Row():
                next_search_button = gr.Button("Next Img", scale=2)
                image_search_button = gr.Button("Search KNN", scale=2)
                convert_ms = gr.Button("Conv Ms", scale=2)
                submit_kis_button = gr.Button("SubKIS", scale=2)
                submit_qa_button = gr.Button("SubQA", scale=2)

            # Second row: Top K Results and Model Type underneath
            with gr.Row():
                top_k = gr.Number(label="Top K Results", value=300, scale=1)
                model_type_input = gr.Dropdown(
                    choices=["clip", "clipv2_l14", "clipv2_h14", "clip + clipv2_l14", "clipv2_l14 + clipv2_h14", "clip + clipv2_h14"],
                    label="Model Type",
                    value="clipv2_l14",  # Thiết lập mặc định nếu cần
                    scale=2
                )
                storage = gr.Dropdown(choices=["local", "cloud"], label="DB", value="cloud", scale=1)
                folder_name = gr.Textbox(label="Folder name", placeholder="L00_V000....", scale=1)
                image_name = gr.Textbox(label="Image name", placeholder="0000....", scale=1)
                milisec_name = gr.Textbox(label="Milisec", scale=1)

                answer = gr.Textbox(label="QA", placeholder="Your answer", scale=1)

            # Third row: Gallery for search results
            image_output = gr.Gallery(label="Search Results", elem_id="image_gallery", columns=5, allow_preview=True, show_fullscreen_button=True)

            # Set search button click action
            search_button.click(fn=text_to_image, 
                                inputs=[text_input, top_k, model_type_input, storage], 
                                outputs=image_output)

            # Correctly passing the function without calling it
            next_search_button.click(fn=search_neighbor,
                                     inputs=[folder_name, image_name, storage],
                                     outputs=image_output)
            
            image_search_button.click(fn=image_search_to_image, inputs=[folder_name, image_name, top_k, storage], outputs=image_output)
            
            convert_ms.click(fn=convert_to_milisecond, inputs = image_name, outputs = milisec_name)
            # Thay đổi trong phần khởi tạo giao diện
            image_output.select(fn=get_folder_select_index, inputs = None, outputs=folder_name)
            image_output.select(fn=get_frame_select_index, inputs = None, outputs=image_name)
            image_output.select(fn=frame_idx_to_milliseconds, inputs = None, outputs=milisec_name)


            submit_kis_button.click(fn=submit_kis_and_vis, 
                                    inputs=[session_id, evaluation_id, folder_name, milisec_name],
                                    outputs=gr.Textbox(label="KIS Submission Result"))

            submit_qa_button.click(fn=submit_qa, 
                                inputs=[session_id, evaluation_id, folder_name, milisec_name, answer],
                                outputs=gr.Textbox(label="QA Submission Result"))
        with gr.Column(scale = 4):
            with gr.Blocks(): 
                with gr.Row():
                    gr.Markdown("## Filter")
                with gr.Row(): 
                    icons_output = gr.Gallery(
                        elem_classes="icon img",
                        value=show_icons(),
                        label="Icons",
                        show_label=False,
                        allow_preview=False)

                    colors_output = gr.Gallery(
                        elem_classes="color img",
                        value=show_colors(),
                        label="Colors",
                        show_label=False,
                        allow_preview=False)
                with gr.Row():
                    tags_input = gr.Textbox(placeholder="Tag", scale=3)
                    color_input = gr.Text(placeholder="Color", scale=3)
                with gr.Row():
                    ocr_input = gr.Textbox(label="OCR Search", placeholder="OCR text")
                    asr_input = gr.Textbox(label="ASR Search", placeholder="ASR text")
                search_button = gr.Button("Search")
                image_output = gr.Gallery(label="Search Results", elem_id="image_gallery", columns=5, allow_preview=True, show_fullscreen_button=True)

# Launch the app
demo.launch(server_port=8386, allowed_paths = [root_db])