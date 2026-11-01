# OpenRouter API Setup Guide

OpenRouter provides free access to multiple LLM models. Here's how to configure it.

---

## Step 1: Get Your API Key

1. Go to [https://openrouter.ai/keys](https://openrouter.ai/keys)
2. Sign up / Log in
3. Create a new API key
4. Copy the key (starts with `sk-or-v1-...`)

---

## Step 2: Set Environment Variable

### Windows (PowerShell)
```powershell
$env:OPENROUTER_API_KEY="sk-or-v1-your-key-here"
```

### Windows (CMD)
```cmd
set OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

### Linux/Mac
```bash
export OPENROUTER_API_KEY="sk-or-v1-your-key-here"
```

---

## Step 3: Choose a Model (Optional)

Set which model to use. Free models are marked with `:free`:

```bash
# Free models (recommended for testing)
export OPENROUTER_MODEL="google/gemini-2.0-flash-001:free"
export OPENROUTER_MODEL="meta-llama/llama-3.1-8b-instruct:free"
export OPENROUTER_MODEL="mistralai/mistral-7b-instruct:free"

# Paid models (better quality, small cost)
export OPENROUTER_MODEL="google/gemini-2.0-flash-001"
export OPENROUTER_MODEL="meta-llama/llama-3.1-8b-instruct"
```

**Recommended free models:**
| Model | Quality | Speed |
|-------|---------|-------|
| `google/gemini-2.0-flash-001:free` | Good | Fast |
| `meta-llama/llama-3.1-8b-instruct:free` | Good | Fast |
| `mistralai/mistral-7b-instruct:free` | OK | Fast |

---

## Step 4: Run the Pipeline

```bash
# Make sure you're in the project directory
cd compliance-doc-intel

# Run with LLM extraction
python run_pipeline.py --method llm --skip-generation

# Run with validation
python run_pipeline.py --method llm --validate --skip-generation
```

---

## Quick Test

Test your API key works:

```python
import os
import httpx

api_key = os.environ.get("OPENROUTER_API_KEY")
model = os.environ.get("OPENROUTER_MODEL", "google/gemini-2.0-flash-001:free")

response = httpx.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    },
    json={
        "model": model,
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 10,
    },
)

print(response.json())
```

---

## Troubleshooting

**Error: "No API key found"**
- Make sure `OPENROUTER_API_KEY` is set in your terminal session
- Restart your terminal after setting the env var

**Error: "Invalid API key"**
- Check the key starts with `sk-or-v1-`
- Regenerate at https://openrouter.ai/keys

**Error: "Model not found"**
- Check the model name at https://openrouter.ai/models
- Use the full model ID (e.g., `google/gemini-2.0-flash-001:free`)

**Slow responses**
- Free models may have rate limits
- Try a different model or wait a moment

---

## Cost

Free models = **$0 cost**

Paid models on OpenRouter are very cheap:
- Gemini Flash: ~$0.075/1M input tokens
- Llama 3.1 8B: ~$0.05/1M input tokens

For 300 documents: **~$0.01-0.10 total**
