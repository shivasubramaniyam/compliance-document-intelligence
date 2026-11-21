# Ollama Local Setup Guide

Run Phase 2 LLM-based extraction 100% locally with Ollama — **no API keys, zero cost, and zero data leaves your machine**.

---

## Hardware Profile & Recommended Models

System detected:
- **CPU**: Intel Core i5-1135G7 (4 cores / 8 threads)
- **RAM**: 18 GB system memory
- **GPU**: NVIDIA GeForce MX350 (2 GB VRAM)

### Model Recommendations

| Model | Size | RAM/VRAM Requirement | Why & When to Use | Status |
|-------|------|----------------------|-------------------|--------|
| **`llama3.2:3b`** *(Recommended)* | ~2.0 GB | ~2.2 GB RAM / partial VRAM | **Sweet spot** for your system: fast, fits memory easily, excellent JSON extraction. | **Installed locally** |
| `qwen2.5:3b` | ~1.9 GB | ~2.1 GB RAM / partial VRAM | Exceptional benchmark performance for structured JSON output. | Run `ollama pull qwen2.5:3b` |
| `llama3.2:1b` | ~1.3 GB | ~1.5 GB VRAM | Fits entirely in your 2GB MX350 VRAM for ultra-fast throughput. | Run `ollama pull llama3.2:1b` |
| `qwen2.5:7b` | ~4.7 GB | ~6.0 GB RAM | Top-tier accuracy, runs on CPU + 18GB RAM (takes ~2-4s per doc). | Run `ollama pull qwen2.5:7b` |

---

## Step 1: Start Ollama Server

Make sure the Ollama server is running in a terminal:

```bash
ollama serve
```

*(If Ollama is already running as a system service or daemon, you can skip this step).*

### Note for system-wide installations:
If you installed Ollama system-wide and your model was saved under `/usr/share/ollama/.ollama/models`, you can make sure Ollama finds it:
```bash
export OLLAMA_MODELS=/usr/share/ollama/.ollama/models
ollama serve
```
Or simply verify:
```bash
ollama list
```
You should see `llama3.2:3b` listed.

---

## Step 2: Test Ollama Connection

You can quickly verify that Ollama is responding and ready to extract JSON:

```bash
python3 -c "
import httpx
res = httpx.post(
    'http://localhost:11434/api/chat',
    json={
        'model': 'llama3.2:3b',
        'messages': [{'role': 'user', 'content': 'Output {\"status\": \"ok\"}'}],
        'stream': False,
        'format': 'json'
    },
    timeout=30.0
)
print('Response:', res.json()['message']['content'])
"
```

---

## Step 3: Run Phase 2 Pipeline

### Run Extraction with Ollama (Default)
No API key required! The pipeline will automatically connect to `http://localhost:11434` with `llama3.2:3b`:

```bash
# LLM Extraction only
python run_pipeline.py --method llm --skip-generation

# LLM Extraction + Compliance Rule Validation
python run_pipeline.py --method llm --validate --skip-generation
```

### Specifying a Different Ollama Model
If you pull another model like `qwen2.5:3b`:
```bash
python run_pipeline.py --method llm --model qwen2.5:3b --validate --skip-generation
```

### Environment Variables (Optional)
You can also control Ollama settings via environment variables:
```bash
export LLM_PROVIDER="ollama"             # 'ollama' or 'openrouter'
export OLLAMA_MODEL="llama3.2:3b"        # Model tag
export OLLAMA_BASE_URL="http://localhost:11434"  # Server URL
```
