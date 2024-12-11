#!/bin/bash

# Define datasets
datasets=(
    "mdattrvuive/keyframeb3 ./dataset/keyframe"
)

# Add video dataset conditionally
add_video_dataset() {
    datasets+=("minhdat13/videos ./dataset/video")
}

# Check if Kaggle API is setup
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "❌ Kaggle API Token was not found. Create the kaggle.json file in ~/.kaggle/ folder"
    exit 1
fi

chmod 600 $HOME/.kaggle/kaggle.json

# Install Kaggle CLI if you don't have it already
if ! command -v kaggle &> /dev/null; then
    echo "⏳ Installing Kaggle CLI..."
    pip install kaggle --quiet
    if [ $? -ne 0 ]; then
        echo "❌ Unable to install Kaggle CLI. Please check your network connection or access rights."
        exit 1
    fi
    echo "✅ Kaggle CLI has been successfully installed."
fi

# Ask user if they want to download the video dataset
echo "❓ Do you want to download the video dataset? (y/n)"
read -r download_video

if [[ "$download_video" =~ ^[Yy]$ ]]; then
    echo "✅ Video dataset will be downloaded."
    add_video_dataset
else
    echo "❌ Skipping video dataset download."
fi

# Function to download dataset
download_dataset() {
    local dataset_id="$1"
    local dataset_dir="$2"

    echo "⏳ Downloading dataset $dataset_id to $dataset_dir..."
    kaggle datasets download -d "$dataset_id" -p "$dataset_dir" --unzip

    if [ $? -eq 0 ]; then
        echo "✅ Download dataset successful: $dataset_dir"
    else
        echo "❌ Have some errors. Please check dataset_id or network"
    fi
}

# Loop through datasets and download them
for dataset in "${datasets[@]}"; do
    dataset_id=$(echo $dataset | cut -d' ' -f1)
    dataset_dir=$(echo $dataset | cut -d' ' -f2)
    download_dataset "$dataset_id" "$dataset_dir"
done
