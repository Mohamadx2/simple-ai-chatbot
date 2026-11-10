# Tiny Python AI Chatbot

A deliberately small command-line chatbot using Hugging Face Transformers and the `sshleifer/tiny-gpt2` model.

The model weights are far below 100 MB and are downloaded automatically the first time the script runs. This is an educational demo, not a capable conversational assistant: the tiny model can produce repetitive or incoherent text because it has very limited capacity and is not instruction-tuned.

## Requirements

- Python 3.10 or newer
- Internet access on the first run, so Hugging Face can download the model
- A few hundred MB of disk space for the Python runtime dependencies; the model itself is tiny

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python chatbot.py
```

Type `quit`, `exit`, or `:q` to stop.

## How it works

The script builds a text-generation pipeline with `sshleifer/tiny-gpt2`, sends the recent conversation as a prompt, and prints a short continuation. The model runs on the CPU by default, so no GPU is required.

## License and model attribution

The chatbot code is released under the MIT License. The model is hosted by [Hugging Face](https://huggingface.co/sshleifer/tiny-gpt2); review the model repository's terms before redistributing model files.
