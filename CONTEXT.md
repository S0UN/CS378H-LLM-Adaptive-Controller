# LLM Adaptive Controller - Project Context

**Last Updated:** February 2026  
**Project Goal:** Build an intelligent system that automatically selects the most efficient LLM quantization level to meet quality targets while minimizing compute costs.

---

## The Problem We're Solving

Running large language models (LLMs) is expensive. Using a high-quality model (Q8_0) for every task wastes compute and money when a smaller model (Q4_K_M) would produce equally good results.

**The Challenge:** How do we automatically determine the minimum viable quantization level for each task type while maintaining quality standards?

**Our Solution:** An adaptive system that:
1. Tests all quantization levels against a gold standard (GPT-4o)
2. Finds the "efficiency floor" (lowest quantization that meets quality threshold)
3. Generates training data for a neural network to learn optimal routing decisions
4. Continuously improves through automated feedback loops

---

## What is Quantization?

Quantization reduces model size by representing weights with fewer bits:

| Quantization | Size (7B model) | Quality | Speed | Use Case |
|-------------|----------------|---------|-------|----------|
| Q2_K | ~2.5 GB | Lowest | Fastest | Testing only |
| Q3_K_M | ~3.3 GB | Low-Medium | Very Fast | Resource-constrained |
| **Q4_K_M** | ~4.1 GB | Good | Fast | **Recommended default** |
| **Q5_K_M** | ~4.8 GB | High | Moderate | Quality-sensitive tasks |
| Q6_K | ~5.5 GB | Very High | Slower | Research/analysis |
| Q8_0 | ~7.2 GB | Excellent | Slowest | Maximum quality needed |

**Key Insight:** A Q4_K_M model can often produce outputs 90% as good as Q8_0 while using 40% less memory and running 2-3x faster.

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Orchestrator                              │
│  • Entry point for the entire system                        │
│  • Coordinates all services                                 │
│  File: orchestrator.py                                      │
└────────────────────┬────────────────────────────────────────┘
                     │
     ┌───────────────┴────────────────┐
     │                                │
┌────▼─────────────────┐   ┌─────────▼──────────────────┐
│  Dataset Generation  │   │   Controller (FastAPI)     │
│  • Inference Loop    │   │   • REST API wrapper       │
│  • Model Manager     │   │   • Talks to llama.cpp     │
│  • Grading Service   │   │   File: controller/main.py │
│  • Result Logging    │   └────────────────────────────┘
└──────────────────────┘
     │
     ├─► Model Downloader     (ModelDownloader class)
     ├─► Dataset Service      (DatasetService class)
     ├─► Inference Loop       (InferenceLoopService class)
     ├─► Grader Service       (GraderService - GPT-4o comparison)
     ├─► Results Logger       (ResultsLoggingService)
     └─► Cache Service        (LRU model caching)

┌─────────────────────────────────────────────────────────────┐
│                Docker Infrastructure                         │
│  • llama.cpp server (serves GGUF models)                    │
│  • Controller API (FastAPI wrapper)                         │
│  File: docker-compose.yml                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Repository Structure & Navigation Guide

### **Root Directory**
```
CS378H-LLM-Adaptive-Controller/
├── orchestrator.py          ← START HERE: Main entry point
├── docker-compose.yml       ← Infrastructure definition
├── requirements.txt         ← Python dependencies
├── Makefile                 ← Common commands
├── .gitignore              ← Excludes models/ folder
├── results.log             ← Output from inference runs
└── CONTEXT.md              ← This file
```

### **controller/**
The FastAPI service that wraps llama.cpp for easier interaction.

```
controller/
├── main.py                  ← FastAPI app with /chat endpoint
├── Dockerfile              ← Container definition
└── requirements.txt        ← (deleted - uses root requirements.txt)
```

**Purpose:** Provides a simple REST API to interact with the llama.cpp model server.

**Key Concepts:**
- Receives chat requests with prompt, max_tokens, temperature
- Forwards to llama.cpp server (running in Docker)
- Returns formatted response

**When to look here:** Understanding the API layer between your code and llama.cpp

---

### **Dataset Generation/**
The core data pipeline for training set generation.

```
Dataset Generation/
├── inference_loop_service.py    ← Main inference orchestration
├── model_manager.py             ← Download & manage GGUF models
├── dataset.py                   ← Load datasets from HuggingFace
├── cache_service.py             ← LRU caching for models
├── logging_service.py           ← Structured logging
├── results_logging_service.py   ← Results persistence
└── model_config.py              ← Model configuration constants
```

#### **Key Files Explained:**

**`inference_loop_service.py`** - MOST IMPORTANT
- The heart of the efficiency floor system
- Runs prompts through multiple quantization levels
- Compares outputs against GPT-4o gold standard
- Finds minimum viable quantization per task
- Generates training data for neural network

**When to read:** Understanding how the system discovers optimal quantizations

**`model_manager.py`**
- Downloads GGUF models from HuggingFace
- Manages model lifecycle (download, cache, delete)
- OOP design with ModelDownloader class

**When to read:** Understanding model acquisition and management

**`dataset.py`**
- Loads conversational datasets from HuggingFace
- Currently supports: Capybara, OpenHermes-2.5
- Provides unified interface for different dataset formats

**When to read:** Understanding input data sources

**`cache_service.py`**
- LRU (Least Recently Used) caching for loaded models
- Keeps hot models in RAM, evicts cold ones
- Memory management for multi-model scenarios

**When to read:** Understanding model switching and memory optimization

**`logging_service.py`**
- Structured logging for debugging and tracing
- Records all system events with timestamps

**When to read:** Debugging or adding new log points

**`results_logging_service.py`**
- Persists inference results to disk
- Tracks: prompt, quantization used, output, quality score, latency

**When to read:** Understanding result storage and analysis

**`model_config.py`**
- Central configuration for model repositories and quantizations
- Default settings for the system

**When to read:** Changing default models or quantization levels

---

### **grader/**
GPT-4o integration for quality assessment.

```
grader/
├── __init__.py
├── agent.py                 ← GraderService class
├── agentTools.py           ← (Placeholder for future tools)
└── prompts.py              ← System prompts for grading
```

**Purpose:** Uses GPT-4o as the "gold standard" to grade outputs from quantized models.

**Key Concepts:**
- Sends both the quantized model output and the expected output to GPT-4o
- GPT-4o judges quality, similarity, correctness
- Returns a score/grade that determines if quantization met threshold

**When to look here:** 
- Understanding quality evaluation logic
- Modifying grading criteria
- Adding new evaluation metrics

---

### **models/**
Storage for downloaded GGUF model files.

```
models/
├── llama-2-7b-chat.Q2_K.gguf
├── llama-2-7b-chat.Q3_K_M.gguf
├── llama-2-7b-chat.Q4_K_M.gguf
├── llama-2-7b-chat.Q5_K_M.gguf
├── llama-2-7b-chat.Q8_0.gguf
└── ... (11 quantization levels downloaded)
```

**Important:** This folder is in `.gitignore` (files too large for GitHub)

**Size:** Each file is 2-7 GB, total ~50 GB for all quantizations

---

### **tests/**
Integration and unit tests.

```
tests/
└── test_integration_agent.py    ← Tests for grading agent
```

**When to look here:** Adding test coverage for new features

---

## How The System Works (End-to-End Flow)

### **Phase 1: Initialization**
1. `orchestrator.py` starts
2. Loads environment variables (dataset name, model repo, quantization)
3. Creates `ModelDownloader` instance
4. Downloads initial model if not cached
5. Starts Docker containers (llama.cpp + controller)

### **Phase 2: Dataset Loading**
1. `DatasetService` fetches conversational dataset from HuggingFace
2. Dataset contains: user prompts + expected assistant responses
3. Example: Capybara has ~15,000 high-quality conversations

### **Phase 3: Inference Loop (The Core)**
For each conversation in the dataset:

1. **Extract prompt and expected output**
   - Parse dataset format (varies by source)
   - Extract the user's question/prompt
   - Store the expected assistant response as gold standard

2. **Run inference across ALL quantization levels**
   - Q2_K → Q3_K_M → Q4_K_M → Q5_K_M → Q6_K → Q8_0
   - For each quantization:
     - Switch model in Docker (via docker-compose restart with MODEL_FILE env var)
     - Send prompt to controller API
     - Receive generated response
     - Record latency and resource usage

3. **Grade each output against GPT-4o**
   - Send to GraderService
   - GPT-4o compares: quantized output vs. expected output
   - Returns similarity score (0-100%)
   - Example: "Q4_K_M output is 92% similar to expected"

4. **Find efficiency floor**
   - Iterate through quantizations from lowest to highest
   - Find first one that meets quality threshold (e.g., ≥85% similarity)
   - Example: Q3_K_M = 78% (fail), Q4_K_M = 92% (pass) → **Floor = Q4_K_M**

5. **Record training data**
   - Store: (prompt, task_type, quality_threshold) → optimal_quantization
   - Example: (summarization_task, 85%_quality) → Q4_K_M
   - Append to results.log

### **Phase 4: Continuous Learning (Future)**
1. Accumulate training dataset over many runs
2. Train neural network on: input features → optimal quantization
3. Deploy neural network as router
4. New requests: NN predicts quantization without testing all levels
5. Periodically re-test to validate NN accuracy

---

## Getting Started - How to Use This Repository

### **1. First Time Setup**

```bash
# Clone repository
cd CS378H-LLM-Adaptive-Controller

# Install Python dependencies
pip install -r requirements.txt

# Set up environment variables
cp secret.env.example secret.env  # if exists, otherwise create it
# Edit secret.env and add:
#   OPENAI_API_KEY=your_key_here
#   DATASET_NAME=Capybara
#   MODEL_QUANT=Q4_K_M

# Download models (optional - orchestrator will download if missing)
cd "Dataset Generation"
python model_manager.py Q4_K_M Q5_K_M Q8_0

# Start Docker infrastructure
docker-compose up -d
```

### **2. Run the Inference Loop**

```bash
# From project root
python orchestrator.py
```

This will:
- Load the Capybara dataset
- Run inference across all quantization levels
- Grade outputs against GPT-4o
- Find efficiency floors
- Log results to `results.log`

### **3. Check Results**

```bash
# View results
cat results.log

# Results format:
# [timestamp] Task: <prompt_preview>
# Q2_K: score=78%, latency=1.2s, status=FAIL
# Q4_K_M: score=92%, latency=2.1s, status=PASS (FLOOR)
# Q8_0: score=96%, latency=4.5s, status=PASS (overkill)
```

---

## Key Environment Variables

Set these in `secret.env` or export in shell:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *required* | OpenAI API key for GPT-4o grader |
| `DATASET_NAME` | "Capybara" | Dataset to use (Capybara, OpenHermes-2.5) |
| `MODEL_REPO` | LLAMA2_7B | HuggingFace repo for GGUF models |
| `MODEL_QUANT` | Q4_K_M | Starting quantization level |
| `MODEL_CACHE_DIR` | ./models | Where to store downloaded models |
| `MODEL_FILE` | (set by system) | Docker env var for active model |
| `CTX` | 4096 | Context window size for llama.cpp |
| `DEBUG_MODE` | INFO | Logging level (DEBUG, INFO, WARNING) |

---

## Understanding the Code - Where to Start

### **If you want to understand...**

**...the overall system flow:**
→ Start with `orchestrator.py` - it coordinates everything

**...how inference works:**
→ Read `Dataset Generation/inference_loop_service.py`
→ Focus on `run_inference_on_turn()` method

**...how models are downloaded:**
→ Read `Dataset Generation/model_manager.py`
→ Look at `ModelDownloader` class

**...how grading works:**
→ Read `grader/agent.py`
→ Look at `GraderService.grade()` method

**...how the API works:**
→ Read `controller/main.py`
→ Simple FastAPI wrapper around llama.cpp

**...how Docker infrastructure works:**
→ Read `docker-compose.yml`
→ Two services: model server (llama.cpp) + controller (FastAPI)

**...how results are stored:**
→ Read `Dataset Generation/results_logging_service.py`
→ Appends to `results.log`

**...how datasets are loaded:**
→ Read `Dataset Generation/dataset.py`
→ Wraps HuggingFace datasets library

---

## Future Enhancements (Roadmap)

### **Phase 1: Current (Data Collection)** [COMPLETED]
- [x] Inference loop across quantizations
- [x] GPT-4o grading
- [x] Efficiency floor detection
- [x] Training data generation
- [x] LRU model caching

### **Phase 2: Neural Network Router** (Next)
- [ ] Feature engineering from prompts
- [ ] Train classifier: input → optimal quantization
- [ ] Deploy NN as routing service
- [ ] A/B test: NN predictions vs. actual floors

### **Phase 3: Production Features**
- [ ] Priority-based task scheduler
- [ ] SLA enforcement (max latency deadlines)
- [ ] Hardware-aware resource management (CPU, memory, thermal)
- [ ] Distributed tracing (OpenTelemetry)
- [ ] Comprehensive telemetry (Prometheus, Grafana)
- [ ] Real-time dashboards

### **Phase 4: Cloud Infrastructure**
- [ ] Deploy to AWS ECS/EKS
- [ ] Auto-scaling based on queue depth
- [ ] S3 model storage
- [ ] CloudWatch monitoring
- [ ] Multi-region deployment

---

## Common Development Tasks

### **Add a new dataset:**
1. Add to `DatasetService.url_map` in `dataset.py`
2. Handle format differences in `parse_turn()` if needed
3. Set `DATASET_NAME` environment variable

### **Add a new quantization level:**
1. Download model: `python model_manager.py <QUANT>`
2. System automatically detects available models
3. Inference loop will test it

### **Change grading criteria:**
1. Modify prompts in `grader/prompts.py`
2. Update `GraderService.grade()` in `grader/agent.py`
3. Adjust quality threshold in inference loop

### **Add new metrics:**
1. Extend `ResultsLoggingService` in `results_logging_service.py`
2. Add fields to log format
3. Update parsing/analysis scripts

### **Debug a failed inference:**
1. Check `results.log` for error details
2. Set `DEBUG_MODE=DEBUG` for verbose logging
3. Check Docker logs: `docker-compose logs model`
4. Test controller directly: `curl http://localhost:3000/chat`

---

## Key Concepts & Terminology

**Quantization Level:** Number of bits used to represent model weights (Q2 = 2-bit, Q8 = 8-bit)

**Efficiency Floor:** The lowest quantization that meets quality threshold for a given task

**Gold Standard:** GPT-4o output used as reference for quality comparison

**LRU Cache:** Least Recently Used cache - keeps hot models loaded, evicts cold ones

**Inference Loop:** Process of running prompts through models and collecting results

**Grading:** Comparing quantized model output to gold standard using GPT-4o

**llama.cpp:** Fast C++ inference engine for GGUF models

**GGUF:** File format for quantized LLaMA models (successor to GGML)

**Docker Compose:** Tool for running multi-container applications (model server + controller)

**HuggingFace:** Platform for ML models and datasets

---

## Troubleshooting

### **"OPENAI_API_KEY not found"**
→ Set in `secret.env` or export: `export OPENAI_API_KEY=sk-...`

### **"Model not found in cache"**
→ Run model downloader: `cd "Dataset Generation" && python model_manager.py Q4_K_M`

### **"Docker container not responding"**
→ Check status: `docker-compose ps`
→ Restart: `docker-compose restart model`
→ Check logs: `docker-compose logs model`

### **"Dataset loading failed"**
→ Check internet connection (HuggingFace downloads)
→ Try different dataset: `export DATASET_NAME=OpenHermes-2.5`

### **"Inference timeout"**
→ Increase timeout in `controller/main.py` (default 60s)
→ Use smaller quantization (faster inference)
→ Reduce max_tokens in request

### **"Out of memory"**
→ Close other applications
→ Use smaller quantization (Q2_K, Q3_K_M)
→ Reduce context window: `export CTX=2048`

---

## Questions to Ask Yourself When Reading Code

1. **What is this component's responsibility?**
   - Each service has a single, clear purpose

2. **What are the inputs and outputs?**
   - Look at method signatures and return types

3. **What could go wrong here?**
   - Check try/except blocks and error handling

4. **Where does the data come from?**
   - Trace backwards to understand data flow

5. **Where does the data go next?**
   - Trace forwards to understand pipeline

6. **Why is this needed?**
   - Every component solves a specific problem

---

## Learning Path for New Contributors

### **Week 1: Understanding the System**
- Read this CONTEXT.md fully
- Run `orchestrator.py` and observe output
- Read `orchestrator.py` code
- Understand Docker setup in `docker-compose.yml`

### **Week 2: Deep Dive on Core Services**
- Study `inference_loop_service.py`
- Study `model_manager.py`
- Study `grader/agent.py`
- Trace a single prompt through the entire system

### **Week 3: Experimentation**
- Try different datasets
- Try different quantization levels
- Modify grading prompts
- Analyze `results.log` data

### **Week 4: Extension**
- Add a new feature (e.g., new metric)
- Implement a small enhancement
- Write tests for your code
- Document your changes

---

## Design Principles

1. **Modularity:** Each service has a single responsibility
2. **OOP:** Classes encapsulate state and behavior
3. **Type Hints:** Every function has clear types
4. **Logging:** Every important event is logged
5. **Configuration:** Behavior controlled by environment variables
6. **Error Handling:** Graceful degradation, not crashes
7. **Documentation:** Code explains "why", not just "what"

---

## Additional Resources

**LLaMA.cpp Documentation:** https://github.com/ggerganov/llama.cpp  
**GGUF Format Spec:** https://github.com/ggerganov/ggml/blob/master/docs/gguf.md  
**HuggingFace Datasets:** https://huggingface.co/docs/datasets  
**Quantization Explained:** https://huggingface.co/docs/transformers/main/en/quantization  
**Docker Compose Guide:** https://docs.docker.com/compose/  

---

## Contributing

When adding new features:
1. Follow existing code style (OOP, type hints, logging)
2. Add docstrings to all classes and methods
3. Update this CONTEXT.md if adding new components
4. Test locally before committing
5. Update environment variable docs if needed

---

## Notes

- This is a research project, not production software
- Prioritize clarity over performance optimizations
- Document your assumptions and design decisions
- Ask questions if something is unclear

---

**Remember:** The goal is not just to build working code, but to create a system that teaches us how to make LLMs more efficient. Every component serves that learning objective.

**Happy coding!**
