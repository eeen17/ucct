0. Install `uv`. `uv` is a fast Python manager. Refer to the [docs](https://docs.astral.sh/uv/guides/install-python/) for installation steps.

1. Run `uv sync`. This installs all necessary packages for you.

2. Run the `main.py` file for each model:
```bash
uv run main.py --model ["llama-3.1-8b-instruct", "phi-4", "qwen3"]
```