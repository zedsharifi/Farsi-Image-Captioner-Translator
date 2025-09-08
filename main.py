# main.py
# This is the main script to run the image captioning and translation pipeline.

import argparse
from captioner import ImageCaptioner
from translator import TranslationModel
import utils

# The Google Drive file ID for the pre-trained ClipCap model weights.
MODEL_WEIGHTS_FILE_ID = "14pXWwB4Zm82rsDdvbGguLfx9F8aM7ovT"
MODEL_WEIGHTS_PATH = "coco_weights.pkl"

def main(image_source: str, display: bool):
    """
    The main function to execute the captioning and translation workflow.

    Args:
        image_source (str): URL or local file path of the image.
        display (bool): Whether to display the image using matplotlib.
    """
    # --- 1. Download Model Weights ---
    utils.download_model_weights(MODEL_WEIGHTS_FILE_ID, MODEL_WEIGHTS_PATH)
    
    # --- 2. Load Image ---
    print(f"Loading image from: {image_source}")
    if image_source.startswith(('http://', 'https://')):
        image = utils.load_image_from_url(image_source)
    else:
        image = utils.load_image_from_path(image_source)

    if image is None:
        print("Could not load image. Exiting.")
        return

    if display:
        utils.display_image(image, title="Input Image")

    # --- 3. Initialize Models ---
    # Initialize the captioner with the path to the downloaded weights
    caption_model = ImageCaptioner(model_path=MODEL_WEIGHTS_PATH)
    
    # Initialize the translation model
    translation_model = TranslationModel()

    # --- 4. Generate Caption ---
    print("\nGenerating caption...")
    english_caption = caption_model.generate_caption(image, use_beam_search=True)
    print(f"  [English Caption]: {english_caption}")

    # --- 5. Translate Caption ---
    print("\nTranslating caption to Farsi...")
    farsi_caption = translation_model.translate(english_caption, src_lang="eng", tgt_lang="pes")
    print(f"  [Farsi Translation]: {farsi_caption}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a caption for an image and translate it to Farsi.")
    parser.add_argument(
        "image_source",
        type=str,
        help="The URL or local file path of the image to process."
    )
    parser.add_argument(
        "--no-display",
        action="store_false",
        dest="display",
        help="Do not display the image using matplotlib."
    )
    
    args = parser.parse_args()
    main(args.image_source, args.display)
