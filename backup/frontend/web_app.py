import gradio as gr
import requests
from PIL import Image
import io
import base64

# Helper function to decode base64 images
def decode_image(image_b64):
    image_data = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(image_data))

import base64
from PIL import Image
import io

# Helper function to decode base64-encoded image
def decode_base64_image(image_b64):
    image_data = base64.b64decode(image_b64)  # Decode base64 string
    return Image.open(io.BytesIO(image_data))  # Convert to PIL Image


# Gradio function to interact with backend
def generate_image(prompt, rating=50):
    url = "url_from_ngrok/generate" # Replace with your ngrok URL
    payload = {
        "prompt": prompt,
        "guidance_scale": 7.5 + (rating / 20),  # Adjust based on rating
        "num_inference_steps": 50,
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        image_b64 = response.json()["image"]
        image = decode_base64_image(image_b64)
        return image
    else:
        return f"Error: {response.status_code}, {response.text}"

# Define Gradio interface
iface = gr.Interface(
    fn=generate_image,
    inputs=[
        "text",  # Text input for the prompt
        gr.Slider(0, 100, step=1, value=50, label="Rate the image"),  # Use gr.Slider here
    ],
    outputs="image",  # Display the generated image
)
# Launch Gradio app
iface.launch()
