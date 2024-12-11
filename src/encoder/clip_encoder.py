import os
import open_clip
import glob
import clip
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
from datetime import datetime

from mlflow_configs.config_schemas.mlflow_schema import MLFlowConfig
from mlflow_configs import mlflow_utils

from .data_path_process import process_data_path

class CLIPencoder():
    def __init__(self, batch_size: int, mlflow_config: MLFlowConfig):
        self.__device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_l14, _, self.preprocess_l14 = open_clip.create_model_and_transforms(
            'ViT-L-14', device=self.__device, pretrained='datacomp_xl_s13b_b90k'
        )
        self.model_h14, _, self.preprocess_h14 = open_clip.create_model_and_transforms(
            'ViT-H-14-CLIPA', device=self.__device, pretrained='datacomp1b'
        )
        self.model_b16, self.preprocess_b16 = clip.load("ViT-B/16", device=self.__device)
        self.batch_size = batch_size
        self.mlflow_config = mlflow_config  
        
    def encode(self, data_path: str, save_dir: str, model: str):
        with mlflow_utils.activate_mlflow(
            experiment_name=self.mlflow_config.experiment_name,
            run_name=self.mlflow_config.run_name,
            run_id=self.mlflow_config.run_id,
            tag=self.mlflow_config.tag
        ) as run:
            mlflow_utils.log_param("batch_size", self.batch_size)
            mlflow_utils.log_param("model_type", model)
            mlflow_utils.log_param("data_path", data_path)
            mlflow_utils.log_param("save_dir", save_dir)

            all_keyframe_paths = process_data_path(data_path)
            savedir = f"{save_dir}/{model}"
            os.makedirs(savedir, exist_ok=True)

            if model == 'l14':
                model, preprocess = self.model_l14, self.preprocess_l14
            elif model == 'h14':
                model, preprocess = self.model_h14, self.preprocess_h14
            elif model == 'b16':
                model, preprocess = self.model_b16, self.preprocess_b16

            for key, video_keyframe_paths in tqdm(all_keyframe_paths.items()):
                video_ids = sorted(video_keyframe_paths.keys())
                
                if not os.path.exists(os.path.join(savedir, key)):
                    os.mkdir(os.path.join(savedir, key))
                
                for video_id in tqdm(video_ids):
                    video_feats = []
                    video_keyframe_path = video_keyframe_paths[video_id]
                    for i in range(0, len(video_keyframe_path), self.batch_size):
                        # Process images in batches
                        images = []
                        image_paths = video_keyframe_path[i:i+self.batch_size]
                        for image_path in image_paths:
                            image = preprocess(Image.open(image_path)).unsqueeze(0)
                            images.append(image)
                        images = torch.cat(images).to(self.__device)

                        with torch.no_grad(), torch.cuda.amp.autocast():
                            image_feats = model.encode_image(images)
                        image_feats /= image_feats.norm(dim=-1, keepdim=True)

                        for b in range(image_feats.shape[0]):
                            video_feats.append(image_feats[b].detach().cpu().numpy().astype(np.float32).flatten())

                    np.save(f'{savedir}/{key}/{video_id}.npy', video_feats)
                    mlflow_utils.log_metric(f"processed_{video_id}", len(video_feats))

            # Log artifacts or other files for reproducibility
            mlflow_utils.log_artifacts_for_reproducibility('./src') #TODO: Fix log_artifacts for main folders
            
            # Log the total number of keyframes processed
            mlflow_utils.log_metric("total_videos", len(all_keyframe_paths))

def main():
    parser = argparse.ArgumentParser(description="CLIP Encoder Script")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the data directory containing keyframes.")
    parser.add_argument('--save_dir', type=str, required=True, help="Directory to save encoded features.")
    parser.add_argument('--model', type=str, required=True, choices=['l14', 'h14', 'b16'], help="Model type to use: 'l14', 'h14', or 'b16'.")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for processing images.")
    parser.add_argument('--experiment_name', type=str, default="Default", help="Experiment name for MLFlow.")
    args = parser.parse_args()

    # Initialize MLFlowConfig và encoder
    mlflow_config = MLFlowConfig(
        experiment_name=args.experiment_name,
        run_name=mlflow_utils.generate_run_name_by_date_time()
    )
    
    encoder = CLIPencoder(batch_size=args.batch_size, mlflow_config=mlflow_config)

    # Start encoding with MLflow tracking
    encoder.encode(data_path=args.data_path, save_dir=args.save_dir, model=args.model)


if __name__ == "__main__":
    main()
