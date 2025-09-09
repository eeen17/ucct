import torch
from unsloth import FastLanguageModel
import argparse
from typing import NamedTuple
from transformers import TextStreamer  # type: ignore
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm


def attachHooks(model, layers, all_layer_outputs):
    num_layers = len(layers)
    
    # --- Step 2: Prepare for Multi-Hook Capture ---
    def get_layer_output_hook(layer_idx):
        def hook(module, input, output):
            # print(f"--- layer: {layer_idx} ---")
            # print(f"tuple length: {len(output)}")
            # for i, o in enumerate(output):
            #     print(f"{i}th output: {o}")
            # print("\n\n")

            # many language models output a tuple that gives additional information ([hidden states, attention weights, kv cache]). We only want the first element
            all_layer_outputs[layer_idx] = output[0].detach()
            # nothing is overwritten, hook only runs once

        return hook

    # Attach a hook to every decoder layer
    hook_handles = []
    for i in range(num_layers):
        target_layer = layers[i]
        handle = target_layer.register_forward_hook(get_layer_output_hook(i))
        hook_handles.append(handle)
    print(f"Attached {len(hook_handles)} hooks to layers 0 through {num_layers - 1}.")


class Task(NamedTuple):
    name: str
    input_header: str
    task: str
    br: str
    examples: str
    input_footer: str


# formats to llama-3.1 template
def formatTasks(model_name: str, suite: list[dict[str, str | list[str]]]) -> list[Task]:
    tasks = []
    
    match model_name:
        case "unsloth/Qwen3-14B":
            for t in suite:
                tasks.append(
                    Task(
                        name=str(t["name"]),
                        input_header="<|im_start|>user\n",
                        task=str(t["instruction"]),
                        br = "\nExamples:\n",
                        examples="\n".join(t["examples"]),
                        input_footer = "<|im_end|>\n<im_start>assistant\n<think>\n\n</think>\n\n"   # disable cot
                    ),
                )
        case "unsloth/gemma-3n-E4B-it":
            for t in suite:
                tasks.append(
                    Task(
                        name=str(t["name"]),
                        input_header="<start_of_turn>user\n",
                        task=str(t["instruction"]),
                        br = "\nExamples:\n",
                        examples="\n".join(t["examples"]),
                        input_footer = "<end_of_turn>\n<start_of_turn>model\n"
                    ),
                )
        case "unsloth/Phi-4":
            for t in suite:
                tasks.append(
                    Task(
                        name=str(t["name"]),
                        input_header="<|im_start|>user<|im_sep|>\n\n",
                        task=str(t["instruction"]),
                        br = "\nExamples:\n",
                        examples="\n".join(t["examples"]),
                        input_footer = "\n\n<|im_end|><|im_start|>assistant<|im_sep|>\n\n"
                    ),
                )
        case "meta-llama/Meta-Llama-3.1-8B-Instruct":
            for t in suite:
                tasks.append(
                    Task(
                        name=str(t["name"]),
                        input_header="<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 July 2024\n\nToday Date: 26 July 2024\n\n<|eot_id|><|start_header_id|>human<|end_header_id|>\n\n",
                        task=str(t["instruction"]),
                        br = "<|eot_id|>\n\nExamples:\n",
                        examples="\n".join(t["examples"]),
                        input_footer = "\n\n<|start_header_id|>assistant<|end_header_id|>\n\n"
                    ),
                )

    return tasks


class Range(NamedTuple):
    lower: int
    upper: int


class Input(NamedTuple):
    task_indices: Range
    examples_indices: Range
    tokenized: list[int]


def processTaskInput(task: Task, tokenizer) -> Input:
    lengths = []
    recon_tokens = []  # reconstructed tokens list
    for i, x in enumerate(task[1:]):
        section_tokens = tokenizer.encode(x)[
            0 if i == 0 else 1 :
        ]  # remove each begin_of_text token except first one
        recon_tokens.extend(section_tokens)
        lengths.append(len(section_tokens))

    indices = []
    lower = 0
    for length in lengths:
        indices.append(Range(lower, length + lower))
        lower += length
    print(indices)
    # input_arr = [input_header, task, "<|eot_id|>", examples, input_footer]
    return Input(
        task_indices=indices[1],
        examples_indices=indices[3],
        tokenized=recon_tokens,
    )


def pairwise_cosine_distance(vectors):
    """
    Calculates the mean pairwise cosine distance for a set of vectors.
    Args:
        vectors: A tensor of shape [N, D] where N is the number of vectors
                 and D is the embedding dimension.
    Returns:
        A single scalar value for the mean pairwise cosine distance.
    """
    if vectors.shape[0] < 2:
        return torch.tensor(0.0, device=vectors.device)

    # Normalize each vector to unit length
    normalized_vectors = torch.nn.functional.normalize(vectors, p=2, dim=1, eps=1e-12)

    # Compute the cosine similarity matrix by matrix-multiplying the normalized vectors
    # Resulting shape: [N, N]
    similarity_matrix = torch.matmul(normalized_vectors, normalized_vectors.T)

    # Cosine distance is 1 - similarity
    distance_matrix = 1 - similarity_matrix

    # We only want the unique pairwise distances, which are in the upper
    # triangle of the matrix (excluding the diagonal).
    # Get the indices of the upper triangle
    upper_triangle_indices = torch.triu_indices(
        distance_matrix.shape[0], distance_matrix.shape[1], offset=1
    )

    # Extract the values and calculate the mean
    pairwise_distances = distance_matrix[
        upper_triangle_indices[0], upper_triangle_indices[1]
    ]

    return torch.mean(pairwise_distances)


class HiddenStatesByLayer(NamedTuple):
    rho_d: list[float]
    d_r: list[float]
    task_vecs: list[torch.Tensor]
    example_vecs: list[torch.Tensor]


def calcHiddenStates(
    hidden_states_by_task: dict[str, HiddenStatesByLayer],
    processedTasks: list[tuple[Task, Input]],
    num_layers: int,
    all_layer_outputs: list[torch.Tensor],
):
    for task, input in processedTasks:
        rho_d_by_layer: list[float] = []
        d_r_by_layer: list[float] = []
        task_vecs_by_layer: list[torch.Tensor] = []
        example_vecs_by_layer: list[torch.Tensor] = []

        for i in range(num_layers):
            layer_hidden_states = all_layer_outputs[i]

            lower, upper = input.task_indices
            task_vecs = layer_hidden_states[0, lower:upper, :]
            lower, upper = input.examples_indices
            example_vecs = layer_hidden_states[0, lower:upper, :]
            # print(task_vecs.shape)
            # print(example_vecs.shape)
            task_vecs_by_layer.append(task_vecs)
            example_vecs_by_layer.append(example_vecs)

            # Calculate Pattern Density (rho_d)
            if len(example_vecs) > 1:
                # example_vecs = example_vecs.to(torch.float32)
                mean_cos_dist = pairwise_cosine_distance(example_vecs)
                rho_d = 1 / (mean_cos_dist + 1e-6)
            else:
                print(
                    "WARNING: example token size 1. If this is what you want, ignore this message."
                )
                rho_d = torch.tensor(0)
            rho_d_by_layer.append(rho_d.item())

            # Calculate Representational Mismatch (d_r)
            pattern_centroid = torch.mean(example_vecs, dim=0)
            # print(pattern_centroid.shape)
            # print()
            task_centroid = torch.mean(task_vecs, dim=0)
            d_r = 1 - torch.nn.functional.cosine_similarity(
                task_centroid, pattern_centroid, dim=0
            )
            d_r_by_layer.append(d_r.item())

        hidden_states_by_task[task.name] = HiddenStatesByLayer(
            rho_d_by_layer, d_r_by_layer, task_vecs_by_layer, example_vecs_by_layer
        )


def graphByTask(
    tasks: list[Task],
    hidden_states_by_task: dict[str, HiddenStatesByLayer],
    num_layers: int,
    output_dir: str,
    model_name: str,
):
    all_rho_d = np.array([res.rho_d for res in hidden_states_by_task.values()])
    all_d_r = np.array([res.d_r for res in hidden_states_by_task.values()])

    # Calculate mean and standard deviation across tasks for each layer
    mean_rho_d = np.mean(all_rho_d, axis=0)
    std_rho_d = np.std(all_rho_d, axis=0)

    mean_d_r = np.mean(all_d_r, axis=0)
    std_d_r = np.std(all_d_r, axis=0)

    # # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    layer_indices = list(range(num_layers))

    # print(len(layer_indices))
    # print(len(mean_rho_d))
    # # Plot Aggregated Pattern Density
    ax1.plot(layer_indices, mean_rho_d, marker="o", linestyle="-", label="Mean")
    ax1.fill_between(
        layer_indices,
        mean_rho_d - std_rho_d,
        mean_rho_d + std_rho_d,
        alpha=0.2,
        label="Std. Dev.",
    )
    ax1.set_title(
        "Aggregated Pattern Density ($\\rho_d$) vs. Layer Index (Across All Tasks)"
    )
    ax1.set_ylabel("Pattern Density (1 / Mean Distance)")
    ax1.legend()
    ax1.grid(True)

    # # Plot Aggregated Representational Mismatch
    ax2.plot(
        layer_indices, mean_d_r, marker="o", linestyle="-", color="r", label="Mean"
    )
    ax2.fill_between(
        layer_indices,
        mean_d_r - std_d_r,
        mean_d_r + std_d_r,
        color="r",
        alpha=0.2,
        label="Std. Dev.",
    )
    ax2.set_title(
        "Aggregated Representational Mismatch ($d_r$) vs. Layer Index (Across All Tasks)"
    )
    ax2.set_ylabel("Mismatch (Cosine Distance)")
    ax2.set_xlabel("Layer Index")
    ax2.legend()
    ax2.grid(True)
    fig.suptitle(f"{model_name.split("/")[1]} Aggregated Hidden Density and Distance Trajectories", fontsize=16, fontweight='bold')


    plt.tight_layout()
    plt.savefig(f"{output_dir}/agg.pdf", bbox_inches="tight")
    # plt.show()
    print("\nGenerating overlayed plots for all tasks...")

    # Create the plot figure
    # Create the plot figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    layer_indices = list(range(num_layers))

    # Use a colormap to get distinct colors for each task
    colors = plt.cm.viridis(np.linspace(0, 1, len(tasks)))  # type: ignore

    # Collect handles for legend
    legend_handles = []

    # Loop through each stored task result and plot it
    for i, (task_name, results) in enumerate(hidden_states_by_task.items()):
        # Get the data for this task
        rho_d_values = results.rho_d
        d_r_values = results.d_r

        # Plot Pattern Density
        (line1,) = ax1.plot(
            layer_indices,
            rho_d_values,
            marker="o",
            markersize=4,
            linestyle="-",
            alpha=0.7,
            color=colors[i],
            label=task_name,
        )

        # Plot Representational Mismatch
        ax2.plot(
            layer_indices,
            d_r_values,
            marker="o",
            markersize=4,
            linestyle="-",
            alpha=0.7,
            color=colors[i],
        )

        # Save one handle for the legend
        legend_handles.append(line1)

    # Configure Plot 1
    ax1.set_title("Overlayed Pattern Density ($\\rho_d$) vs. Layer Index")
    ax1.set_ylabel("Pattern Density (1 / Mean Distance)")
    ax1.grid(True)

    # Configure Plot 2
    ax2.set_title("Overlayed Representational Mismatch ($d_r$) vs. Layer Index")
    ax2.set_ylabel("Mismatch (Cosine Distance)")
    ax2.set_xlabel("Layer Index")
    ax2.grid(True)

    # Shared legend outside the figure
    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.8, 0.5),
        title="Task",
        frameon=False,
    )
    fig.suptitle(f"{model_name.split("/")[1]} Overlayed Hidden Density and Distance Trajectories", fontsize=16, fontweight='bold')
    
    # Adjust layout to make room for legend
    plt.tight_layout()
    plt.subplots_adjust(right=0.8)
    plt.savefig(f"{output_dir}/overlay.pdf", bbox_inches="tight")
    # plt.show()


def main(model_name: str, output_dir: str):
    output_dir = f"{output_dir}/{model_name.replace("/", "_")}"
    try:
        os.mkdir(output_dir)
        print(f"Directory '{output_dir}' created successfully.")
    except FileExistsError:
        print(f"Directory '{output_dir}' created successfully.")

    print("--- Setting up Model and Tokenizer with Unsloth ---")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,  # Recommended to set for Unsloth
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )

    if model_name != "unsloth/Qwen3-14B":
        _ = FastLanguageModel.for_inference(model)

    layers = model.model.language_model.layers if model_name == "unsloth/gemma-3n-E4B-it" else model.model.layers
    
    if model_name == "unsloth/gemma-3n-E4B-it":
        tokenizer.encode = lambda x : tokenizer(None, x)['input_ids'][0]
    
    num_layers = len(layers)
    all_layer_outputs = [torch.tensor(0) for _ in range(num_layers)]
    attachHooks(model, layers, all_layer_outputs)

    with open('task_suite.json', 'r') as f:
    # with open('task_suite_test.json', 'r') as f:
        task_suite = json.load(f)
    tasks = formatTasks(model_name, task_suite)
    processedTasks: list[tuple[Task, Input]] = []
    for task in tasks:
        processedTasks.append((task, processTaskInput(task, tokenizer)))

    task_outputs: dict[str, list[torch.Tensor]] = {}
    text_streamer = TextStreamer(tokenizer)

    print(f"{'=' * 10} GENERATING {'=' * 10}")
    for task, input in tqdm(processedTasks):
        print(f"\n\n{'-' * 10} {task.name} {'-' * 10}\n")
        _ = model.generate(
            input_ids=torch.tensor([input.tokenized]).to("cuda"),
            streamer=text_streamer,
            max_new_tokens=256,
            use_cache=True,
        )
        task_outputs[task.name] = all_layer_outputs.copy()

    hidden_states_by_task: dict[str, HiddenStatesByLayer] = {}

    calcHiddenStates(
        hidden_states_by_task, processedTasks, num_layers, all_layer_outputs
    )
    
    print(f"{'=' * 10} WRITING RESULTS {'=' * 10}")
    graphByTask(tasks, hidden_states_by_task, num_layers, output_dir, model_name)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="VisualizeHiddenStates")
    parser.add_argument(
        "--model_name", type=str, choices=["llama-3.1-8b-instruct", "phi-4", "gemma-3n", "qwen3"], default="llama-3.1-8b-instruct"
    )
    parser.add_argument("--output_dir", type=str, default="./output")

    args = parser.parse_args()

    match str(args.model_name).lower():
        case "qwen3":
            model_name = "unsloth/Qwen3-14B"
        case "gemma-3n":
            model_name = "unsloth/gemma-3n-E4B-it"
        case "phi-4":
            model_name = "unsloth/Phi-4"
        case _:
            model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    
    main(model_name, args.output_dir)
