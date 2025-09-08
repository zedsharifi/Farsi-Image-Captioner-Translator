# captioner.py
# This file contains the implementation of the ClipCap model for image captioning.

import torch
from torch import nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
import pickle
from typing import Tuple, List, Union, Optional

# Define the device to be used
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) for mapping embeddings.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class ClipCap(nn.Module):
    """
    The main ClipCap model which combines CLIP embeddings with a language model.
    """
    def __init__(self, prefix_length: int, prefix_size: int = 512):
        super(ClipCap, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        self.clip_project = MLP(
            (prefix_size, (self.gpt_embedding_size * prefix_length) // 2, self.gpt_embedding_size * prefix_length)
        )

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if mask is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            mask = torch.cat((dummy_token, mask), dim=1)
        
        outputs = self.gpt(inputs_embeds=embedding_cat, attention_mask=mask, labels=labels)
        return outputs


class ImageCaptioner:
    """
    A wrapper class to handle image preprocessing and caption generation.
    """
    def __init__(self, model_path: str, prefix_length: int = 10):
        """
        Initializes the captioner.
        
        Args:
            model_path (str): Path to the pre-trained model weights (.pkl file).
            prefix_length (int): The length of the prefix for the language model.
        """
        self.prefix_length = prefix_length
        self.model = ClipCap(prefix_length)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model = self.model.to(DEVICE)
        self.model.eval()
        
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        # GPT2 imports needed for ClipCap model definition
        from transformers import GPT2Tokenizer, GPT2LMHeadModel
        global GPT2LMHeadModel # Make it accessible to the ClipCap class
        GPT2LMHeadModel = GPT2LMHeadModel
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


    def generate_caption(self, image, use_beam_search: bool = False) -> str:
        """
        Generates a caption for a given image.

        Args:
            image (PIL.Image): The input image.
            use_beam_search (bool): Whether to use beam search for generation.

        Returns:
            str: The generated caption.
        """
        with torch.no_grad():
            inputs = self.clip_processor(images=image, return_tensors="pt").to(DEVICE)
            prefix = self.clip_model.get_image_features(**inputs)
            prefix_embed = self.model.clip_project(prefix).reshape(1, self.prefix_length, -1)
            
            if use_beam_search:
                return self._generate_beam(prefix_embed)
            else:
                return self._generate_greedy(prefix_embed)

    def _generate_greedy(self, prefix_embed: torch.Tensor) -> str:
        """Generates a caption using greedy search."""
        model = self.model
        tokenizer = self.tokenizer
        
        # Generate the caption
        output = model.gpt.generate(inputs_embeds=prefix_embed, max_length=20, do_sample=False)
        caption = tokenizer.decode(output[0], skip_special_tokens=True)
        return caption.strip()

    def _generate_beam(self, prefix_embed: torch.Tensor, beam_size: int = 5) -> str:
        """Generates a caption using beam search."""
        # This is a simplified implementation. A full beam search would be more complex.
        model = self.model
        tokenizer = self.tokenizer

        # Generate with beam search
        output = model.gpt.generate(
            inputs_embeds=prefix_embed,
            max_length=20,
            num_beams=beam_size,
            early_stopping=True
        )
        caption = tokenizer.decode(output[0], skip_special_tokens=True)
        return caption.strip()
