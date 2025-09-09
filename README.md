0. Install `uv`. `uv` is a fast Python manager. Refer to the [docs](https://docs.astral.sh/uv/guides/install-python/) for installation steps.

1. Run `uv sync`. This installs all necessary packages for you.

2. `uv run main.py --model_name ...`

3. make sure to specify gpu using `CUDA_VISIBLE_DEVICES=<gpu>` if your cluster has multiple gpus for best performance
