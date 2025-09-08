# utils.py
# This file contains helper functions for loading images and managing files.

import requests
from PIL import Image
from io import BytesIO
import os
import gdown
import matplotlib.pyplot as plt

def load_image_from_url(url: str) -> Image.Image:
    """
    Loads an image from a given URL.
    
    Args:
        url (str): The URL of the image.
        
    Returns:
        PIL.Image.Image: The loaded image object.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return image
    except requests.exceptions.RequestException as e:
        print(f"Error fetching image from URL: {e}")
        return None

def load_image_from_path(path: str) -> Image.Image:
    """
    Loads an image from a local file path.
    
    Args:
        path (str): The local path to the image file.
        
    Returns:
        PIL.Image.Image: The loaded image object.
    """
    try:
        image = Image.open(path)
        return image
    except FileNotFoundError:
        print(f"Error: Image file not found at {path}")
        return None
    except IOError:
        print(f"Error: Could not open image file at {path}")
        return None

def display_image(image: Image.Image, title: str = ""):
    """
    Displays an image using matplotlib.

    Args:
        image (PIL.Image.Image): The image to display.
        title (str): The title for the image plot.
    """
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

def download_model_weights(file_id: str, output_path: str):
    """
    Downloads a file from Google Drive using its file ID.

    Args:
        file_id (str): The Google Drive file ID.
        output_path (str): The path to save the downloaded file.
    """
    if not os.path.exists(output_path):
        print(f"Downloading model weights to {output_path}...")
        gdown.download(id=file_id, output=output_path, quiet=False)
        print("Download complete.")
    else:
        print(f"Model weights already exist at {output_path}.")
