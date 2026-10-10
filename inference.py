# src/inference.py
import torch
from transformers import pipeline

def load_image_classification_pipeline(model_id: str):
    """
    Carga y retorna un pipeline de clasificación de imágenes de Hugging Face
    utilizando GPU si está disponible.
    """
    device = 0 if torch.cuda.is_available() else -1
    print(f"Cargando modelo {model_id} en dispositivo (GPU ID: {device})...")
    
    image_pipe = pipeline(
        "image-classification", 
        model=model_id, 
        device=device
    )
    return image_pipe

def run_inference(pipe, image):
    """
    Ejecuta la inferencia sobre una imagen dada utilizando el pipeline cargado.
    """
    results = pipe(image)
    return results