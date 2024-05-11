import os
import zipfile

from pathlib import Path

import requests

current_file_path = os.path.realpath(__file__)
parent_dir = os.path.dirname(current_file_path)
path = os.path.dirname(parent_dir)

# Setup path to data folder
data_path = Path(path) / "data/"
image_path = data_path / "pizza_steak_sushi"


def download_data(image_path):
    """Download the pizza, steak and sushi images"""
    # If the image folder doesn't exist, download it and prepare it...
    if image_path.is_dir():
        print(f"{image_path} directory exists.")
    else:
        print(f"Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)

        # Download pizza, steak, sushi data
        with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
            request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
            print("Downloading pizza, steak, sushi data...")
            f.write(request.content)

        # Unzip pizza, steak, sushi data
        with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
            print("Unzipping pizza, steak, sushi data...")
            zip_ref.extractall(image_path)

        # Remove zip file
        os.remove(data_path / "pizza_steak_sushi.zip")