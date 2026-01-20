# API Documentation

## Base URL

```
http://localhost:5000
```

Or your ngrok URL when deployed:
```
https://your-ngrok-id.ngrok-free.app
```

## Endpoints

### Health Check

Check if the backend is running and the model is loaded.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "device": "cuda",
  "model_loaded": true
}
```

**Status Codes:**
- `200 OK`: Backend is healthy

---

### Generate Image

Generate an image from a text prompt.

**Endpoint:** `POST /generate`

**Request Body:**
```json
{
  "prompt": "A golden retriever playing in the snow",
  "guidance_scale": 7.5,
  "num_inference_steps": 50
}
```

**Parameters:**

| Parameter | Type | Required | Default | Range | Description |
|-----------|------|----------|---------|-------|-------------|
| `prompt` | string | Yes | - | 1-500 chars | Text description of the image |
| `guidance_scale` | float | No | 7.5 | 1.0-20.0 | How closely to follow the prompt |
| `num_inference_steps` | integer | No | 50 | 10-100 | Number of denoising steps |

**Response:**
```json
{
  "image": "base64_encoded_image_data",
  "prompt": "A golden retriever playing in the snow",
  "guidance_scale": 7.5,
  "num_inference_steps": 50
}
```

**Status Codes:**
- `200 OK`: Image generated successfully
- `400 Bad Request`: Invalid parameters
- `500 Internal Server Error`: Generation failed
- `503 Service Unavailable`: Model not loaded

**Error Response:**
```json
{
  "error": "Error message description"
}
```

---

## Parameter Guide

### Guidance Scale

Controls how closely the generated image follows the prompt.

- **Low (1.0-5.0)**: More creative, less literal interpretation
- **Medium (5.0-10.0)**: Balanced between creativity and accuracy
- **High (10.0-20.0)**: Very literal, closely follows prompt

**Recommendation:** Start with 7.5 and adjust based on results.

### Inference Steps

Number of denoising steps in the diffusion process.

- **Low (10-30)**: Faster generation, lower quality
- **Medium (30-60)**: Good balance of speed and quality
- **High (60-100)**: Best quality, slower generation

**Recommendation:** Use 50 steps for good quality-speed tradeoff.

---

## Example Usage

### Python (requests)

```python
import requests
import base64
from PIL import Image
import io

# Generate image
url = "http://localhost:5000/generate"
payload = {
    "prompt": "A sunset over mountains",
    "guidance_scale": 7.5,
    "num_inference_steps": 50
}

response = requests.post(url, json=payload)
data = response.json()

# Decode and save image
image_data = base64.b64decode(data['image'])
image = Image.open(io.BytesIO(image_data))
image.save('output.png')
```

### cURL

```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A sunset over mountains",
    "guidance_scale": 7.5,
    "num_inference_steps": 50
  }'
```

### JavaScript (fetch)

```javascript
const generateImage = async () => {
  const response = await fetch('http://localhost:5000/generate', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      prompt: 'A sunset over mountains',
      guidance_scale: 7.5,
      num_inference_steps: 50
    })
  });
  
  const data = await response.json();
  
  // Convert base64 to image
  const img = document.createElement('img');
  img.src = 'data:image/png;base64,' + data.image;
  document.body.appendChild(img);
};
```

---

## Rate Limiting

Currently, there are no rate limits, but generation is naturally limited by GPU processing time (~30-60 seconds per image).

## Best Practices

1. **Prompt Engineering**
   - Be specific and descriptive
   - Include style keywords (e.g., "photorealistic", "oil painting")
   - Mention quality (e.g., "high quality", "detailed", "8k")

2. **Parameter Tuning**
   - Start with defaults (guidance=7.5, steps=50)
   - Increase guidance for more literal results
   - Increase steps for better quality

3. **Error Handling**
   - Always check response status codes
   - Handle timeout errors (requests can take 30-60s)
   - Implement retry logic for transient failures

4. **Performance**
   - Batch requests are not supported (one image per request)
   - Consider reducing steps for faster iteration during testing
   - Use lower guidance_scale for creative exploration

---

## Troubleshooting

### Common Errors

**"GPU out of memory"**
- Reduce `num_inference_steps` to 30-40
- Ensure no other processes are using the GPU
- Consider using a GPU with more VRAM

**"Cannot connect to backend"**
- Check if backend is running: `curl http://localhost:5000/health`
- Verify the correct URL and port
- Check firewall settings

**"Request timeout"**
- Increase request timeout (default: 120s)
- Reduce `num_inference_steps`
- Check GPU utilization

---

## Model Information

- **Base Model**: Stable Diffusion XL 1.0
- **Fine-tuning**: LoRA (Low-Rank Adaptation)
- **LoRA Weights**: Available on Hugging Face Hub
- **Output Resolution**: 1024x1024 pixels
- **Output Format**: PNG (base64 encoded)

---

## Version History

### v1.0.0 (Current)
- Initial release
- SDXL with LoRA support
- Basic generation endpoint
- Health check endpoint