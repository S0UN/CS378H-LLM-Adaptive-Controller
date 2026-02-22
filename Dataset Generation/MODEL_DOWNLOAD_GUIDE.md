ca# Model Download Guide

This guide explains how to download quantized GGUF models for your LLM controller.

## What is Quantization?

Quantization reduces model size by using fewer bits to represent weights:
- **Lower bits** (Q2, Q3, Q4) = Smaller files, faster inference, lower quality
- **Higher bits** (Q5, Q6, Q8) = Larger files, slower inference, higher quality

Think of it like image compression: Q2 is like a heavily compressed JPEG, Q8 is like a high-quality PNG.

## Quick Start

### 1. Install Requirements

```bash
cd "Dataset Generation"
pip install huggingface-hub
```

### 2. View Available Quantizations

```bash
python model_manager.py
```

This shows all 18 quantization levels from Q2_K (smallest) to F32 (largest).

### 3. Download Models

**Download one model:**
```bash
python model_manager.py Q4_K_M
```

**Download multiple models:**
```bash
python model_manager.py Q2_K Q4_K_M Q5_K_M Q8_0
```

Models will be saved to `../models/` folder (the project's models directory).

## Recommended Downloads

### Option 1: Quick Test (one model)
```bash
python model_manager.py Q4_K_M
```
- Size: ~4GB
- Quality: Good balance (recommended)
- Best for: Most use cases

### Option 2: Comparison Set (4 models)
```bash
python model_manager.py Q2_K Q4_K_M Q5_K_M Q8_0
```
- Total Size: ~19GB
- Purpose: Compare quality vs performance across different quantizations
- Best for: Research and finding optimal quantization

### Option 3: High Quality Only (2 models)
```bash
python model_manager.py Q5_K_M Q8_0
```
- Total Size: ~12GB
- Purpose: High-quality inference
- Best for: Production use where quality matters

## All Quantization Levels

| Level | Size (7B model) | Quality | Use Case |
|-------|----------------|---------|----------|
| Q2_K | ~2.5GB | Lowest | Testing only |
| Q3_K_M | ~3.3GB | Low | Resource-constrained |
| Q4_K_M * | ~4GB | Good | **Recommended** |
| Q5_K_M * | ~5GB | High | **Quality priority** |
| Q6_K | ~6GB | Very High | Research |
| Q8_0 | ~7GB | Excellent | Reference quality |
| F16 | ~13GB | Perfect | Rarely available |

## Using the Downloaded Models

After downloading, update `docker-compose.yml` to use your chosen model:

```yaml
services:
  model:
    image: ghcr.io/ggml-org/llama.cpp:server
    ports:
      - "8080:8080"
    volumes:
      - ./models:/models:ro
    command: >
      -m /models/llama-2-7b-chat.Q4_K_M.gguf
      -c 4096
      --host 0.0.0.0
      --port 8080
```

Then start your controller:
```bash
docker-compose up
```

## Changing Models

The current script downloads **Llama-2-7B-Chat** models. To use a different model:

1. Open `model_manager.py`
2. Change line 7-8:
   ```python
   MODEL_REPO = "TheBloke/YourModel-GGUF"  # Change this
   MODEL_BASE_NAME = "your-model-name"      # Change this
   ```
3. Run the script again

Find more GGUF models on HuggingFace: https://huggingface.co/models?search=GGUF

## Troubleshooting

**"Error downloading Q2_K_S: 404"**
- Not all quantizations are available for every model
- Try the common ones: Q4_K_M, Q5_K_M, Q6_K, Q8_0

**Download is slow**
- Large files (4-7GB) take time
- Use a stable internet connection
- Downloads resume automatically if interrupted

**Out of disk space**
- Each model is 2-7GB
- Delete old models from `../models/` folder
- Use fewer quantizations

## Tips

1. **Start with Q4_K_M** - It's the sweet spot for most use cases
2. **Compare 2-3 levels** - Download Q4_K_M and Q8_0 to see quality difference
3. **Monitor RAM usage** - Higher quantization = more RAM needed
4. **Not all exist** - Some models only have 4-6 quantization levels available

## Questions?

- What's the difference between Q4_0 and Q4_K_M? **Q4_K_M is newer and better**
- Which should I use? **Q4_K_M for most cases, Q8_0 for best quality**
- Can I delete old models? **Yes, just remove files from ../models/ folder**
- How do I switch models? **Update the filename in docker-compose.yml**
