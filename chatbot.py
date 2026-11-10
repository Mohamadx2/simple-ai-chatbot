"""A tiny command-line AI chatbot powered by sshleifer/tiny-gpt2."""

from __future__ import annotations

from transformers import pipeline

MODEL_ID = "sshleifer/tiny-gpt2"
MAX_CONTEXT_CHARS = 1_200


def build_generator():
    """Load the small text-generation model from Hugging Face."""
    return pipeline(
        task="text-generation",
        model=MODEL_ID,
        device=-1,  # CPU, so the script works without a GPU.
    )


def reply(generator, history: list[str], message: str) -> str:
    """Generate a short reply using the recent conversation as context."""
    prompt = "\n".join(history + [f"User: {message}", "Bot:"])[-MAX_CONTEXT_CHARS:]
    result = generator(
        prompt,
        max_new_tokens=40,
        do_sample=True,
        temperature=0.9,
        top_k=40,
        top_p=0.95,
        repetition_penalty=1.15,
        pad_token_id=generator.tokenizer.eos_token_id,
        return_full_text=False,
    )[0]["generated_text"]

    # Keep the answer compact and avoid accidentally printing a new speaker turn.
    answer = result.split("\n", 1)[0].strip()
    if "User:" in answer:
        answer = answer.split("User:", 1)[0].strip()
    return answer or "I do not have a useful answer yet."


def main() -> None:
    print(f"Loading {MODEL_ID} (downloaded automatically on first run)...")
    generator = build_generator()
    print("TinyBot is ready. Type 'quit' or 'exit' to stop.\n")

    history: list[str] = []
    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not message:
            continue
        if message.lower() in {"quit", "exit", ":q"}:
            print("Goodbye!")
            break

        answer = reply(generator, history, message)
        print(f"Bot: {answer}\n")
        history.extend([f"User: {message}", f"Bot: {answer}"])
        # Keep memory bounded for this tiny demo.
        history = history[-8:]


if __name__ == "__main__":
    main()
