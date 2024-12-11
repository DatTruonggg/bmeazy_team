# Text to Image Retrieval

---
## **Table of Contents**

1. [Setup](#1-setup)
2. [Data preparation](#2-data-preparation)
    - [2.1 Download dataset](#21-download-dataset)
    - [2.2 Local database](#22-local-database)
    - [2.3 Google Cloud Storage](#23-google-cloud-storage)
3. [Docker](#3-docker)
4. [CI/CD using Jenkins](#4-cicd-using-jenkins)


---
## 1. Setup


## 2. Data preparation
### 2.1 Kaggle dataset
- You can get this project's dataset: [link](https://www.kaggle.com/datasets/mdattrvuive/keyframeb3/data)
- EDA or training in Kaggle notebook

### 2.2 Download dataset (Optional)
- The data of this project is very large, about ~200GiB (Keyframe dataset); if you want to extract from scratch, you can download the Video dataset (~200Gib also). But I just got 1/3 of the dataset for this demo. You can run `download_data.sh` to download the dataset.
- If you don't want to download **Video data folder**, please `n` when run bash script.
When you run `download_data.sh`: 

![download_keyframe](./static/images/keyframe-download.png)

## 2.3 Run Clip-encoder (optional)

![clipencoder](./static/images/clip_encoder.png)



![gcp-init](./static/images/gcp-project-init.png)



![dvc-dev](./static/images/dev-to-git-dev.png)

![dvc-to-gcs](./static//images/dvc-to-gcs.png)
