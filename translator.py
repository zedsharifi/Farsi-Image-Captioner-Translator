# translator.py
# This file contains the implementation of the translation model using SeamlessM4T.

import torch
from transformers import AutoProcessor, SeamlessM4TModel

class TranslationModel:
    """
    A wrapper class for the SeamlessM4T translation model.
    """
    def __init__(self, model_name="facebook/seamless-m4t-v2-large"):
        """
        Initializes the translator by loading the model and processor.

        Args:
            model_name (str): The name of the pre-trained model to use from Hugging Face.
        """
        print("Loading translation model...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = SeamlessM4TModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print("Translation model loaded.")

    def translate(self, text_input: str, src_lang: str = "eng", tgt_lang: str = "pes") -> str:
        """
        Translates text from a source language to a target language.

        Args:
            text_input (str): The text to be translated.
            src_lang (str): The source language code (e.g., "eng" for English).
            tgt_lang (str): The target language code (e.g., "pes" for Farsi).

        Returns:
            str: The translated text.
        """
        text_inputs = self.processor(text=text_input, src_lang=src_lang, return_tensors="pt").to(self.device)
        output_tokens = self.model.generate(**text_inputs, tgt_lang=tgt_lang, generate_speech=False)
        
        # Detokenize and clean the output
        translated_text = self.processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
        return translated_text
