# Instructions

Install `uv`. `uv` is a fast Python manager. Refer to the [docs](https://docs.astral.sh/uv/guides/install-python/) for installation steps.

Run the `main.py` file for each model. `uv` will install the correct packages for you.
```bash
uv run main.py --model ["llama-3.1-8b-instruct", "phi-4", "qwen3"]
```
- Graphs are automatically generated in `./output/<model_name>/<graph>.pdf`
- $\rho_d$ and $d_r$ `pd.DataFrame`s are stored as hdf5 files