import os
import json
import base64
import io
from PIL import Image
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

def model_fn(model_dir, context=None):
    """Load and return the model from the specified directory."""
    model = SentenceTransformer(model_dir)
    return model

def input_fn(request_body, request_content_type, context=None):
    """Parse and preprocess the input data from the request."""
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        inputs = input_data.get('inputs', {})

        # Handle base64-encoded image
        if 'image' in inputs:
            try:
                image_data = base64.b64decode(inputs['image'])
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
                return image
            except Exception as e:
                raise ValueError(f"Invalid image data: {e}")

        # Handle plain text input
        elif 'text' in inputs:
            return inputs['text']

        else:
            raise ValueError("JSON must contain either 'image' or 'text' under 'inputs'")
    
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model, context=None):
    """Generate embeddings for either text or image input."""
    if isinstance(input_data, Image.Image):
        embeddings = model.encode(input_data)
    elif isinstance(input_data, str):
        embeddings = model.encode(input_data)
    else:
        raise ValueError("Unsupported input data type. Must be image or text.")
    
    return embeddings.tolist()

def output_fn(prediction, content_type):
    """Serialize the prediction output to the desired content type."""
    if content_type == 'application/json':
        embedding_array = np.array(prediction)
        norm = np.linalg.norm(embedding_array)
        if norm == 0:
            raise ValueError("Embedding norm is zero; cannot normalize.")
        normalized = embedding_array / norm

        final_response = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": normalized.tolist()
                }
            ],
            "model": "sentence-transformers/clip-ViT-B-32",
            "usage": {
                "prompt_tokens": 0, 
                "total_tokens": 0
            }             
        }
        return json.dumps(final_response)
    raise ValueError(f"Unsupported content type: {content_type}")