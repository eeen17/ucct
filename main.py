import torch
from unsloth import FastLanguageModel
import argparse
from typing import NamedTuple
from transformers import TextStreamer  # type: ignore

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import json

from tqdm import tqdm


def attachHooks(model, layers, all_layer_outputs):
    num_layers = len(layers)
    
    def get_layer_output_hook(layer_idx):
        def hook(module, input, output):
            all_layer_outputs[layer_idx] = output[0].detach()
        return hook

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
    recon_tokens = []
    for i, x in enumerate(task[1:]):
        section_tokens = tokenizer.encode(x)[
            0 if i == 0 else 1 :
        ]
        recon_tokens.extend(section_tokens)
        lengths.append(len(section_tokens))

    indices = []
    lower = 0
    for length in lengths:
        indices.append(Range(lower, length + lower))
        lower += length

    return Input(
        task_indices=indices[1],
        examples_indices=indices[3],
        tokenized=recon_tokens,
    )


def pairwise_cosine_distance(vectors):
    if vectors.shape[0] < 2:
        return torch.tensor(0.0, device=vectors.device)

    normalized_vectors = torch.nn.functional.normalize(vectors, p=2, dim=1, eps=1e-12)
    similarity_matrix = torch.matmul(normalized_vectors, normalized_vectors.T)
    distance_matrix = 1 - similarity_matrix

    upper_triangle_indices = torch.triu_indices(
        distance_matrix.shape[0], distance_matrix.shape[1], offset=1
    )

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
            task_vecs_by_layer.append(task_vecs)
            example_vecs_by_layer.append(example_vecs)

            if len(example_vecs) > 1:
                mean_cos_dist = pairwise_cosine_distance(example_vecs)
                rho_d = 1 / (mean_cos_dist + 1e-6)
            else:
                print(
                    "WARNING: example token size 1. If this is what you want, ignore this message."
                )
                rho_d = torch.tensor(0)
            rho_d_by_layer.append(rho_d.item())

            pattern_centroid = torch.mean(example_vecs, dim=0)
            task_centroid = torch.mean(task_vecs, dim=0)
            d_r = 1 - torch.nn.functional.cosine_similarity(
                task_centroid, pattern_centroid, dim=0
            )
            d_r_by_layer.append(d_r.item())

        hidden_states_by_task[task.name] = HiddenStatesByLayer(
            rho_d_by_layer, d_r_by_layer, task_vecs_by_layer, example_vecs_by_layer
        )

def saveTasks(hidden_states_by_task: dict[str, HiddenStatesByLayer], model_name: str, output_dir: str):
    rho_series = []
    d_series = []
    for i, (task_name, results) in enumerate(hidden_states_by_task.items()):
        rho_series.append(pd.Series(results.rho_d, name=task_name))
        d_series.append(pd.Series(results.d_r, name=task_name))
    
    df_rho = pd.concat(rho_series, axis=1)
    df_d = pd.concat(d_series, axis=1)
    
    df_rho.to_hdf(f"{output_dir}/rho_d.h5", key='df', mode='w')
    df_d.to_hdf(f"{output_dir}/d_r.h5", key='df', mode='w')

def graphByTask(
    tasks: list[Task],
    hidden_states_by_task: dict[str, HiddenStatesByLayer],
    num_layers: int,
    model_name: str,
    output_dir: str,
):
    all_rho_d = np.array([res.rho_d for res in hidden_states_by_task.values()])
    all_d_r = np.array([res.d_r for res in hidden_states_by_task.values()])

    mean_rho_d = np.mean(all_rho_d, axis=0)
    std_rho_d = np.std(all_rho_d, axis=0)

    mean_d_r = np.mean(all_d_r, axis=0)
    std_d_r = np.std(all_d_r, axis=0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    layer_indices = list(range(num_layers))

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
    print("\nGenerating overlayed plots for all tasks...")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    layer_indices = list(range(num_layers))

    colors = plt.cm.viridis(np.linspace(0, 1, len(tasks)))  # type: ignore

    legend_handles = []

    for i, (task_name, results) in enumerate(hidden_states_by_task.items()):
        rho_d_values = results.rho_d
        d_r_values = results.d_r

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

        ax2.plot(
            layer_indices,
            d_r_values,
            marker="o",
            markersize=4,
            linestyle="-",
            alpha=0.7,
            color=colors[i],
        )

        legend_handles.append(line1)

    ax1.set_title("Overlayed Pattern Density ($\\rho_d$) vs. Layer Index")
    ax1.set_ylabel("Pattern Density (1 / Mean Distance)")
    ax1.grid(True)

    ax2.set_title("Overlayed Representational Mismatch ($d_r$) vs. Layer Index")
    ax2.set_ylabel("Mismatch (Cosine Distance)")
    ax2.set_xlabel("Layer Index")
    ax2.grid(True)

    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.8, 0.5),
        title="Task",
        frameon=False,
    )
    fig.suptitle(f"{model_name.split("/")[1]} Overlayed Hidden Density and Distance Trajectories", fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.8)
    plt.savefig(f"{output_dir}/overlay.pdf", bbox_inches="tight")


def main(model_name: str, output_dir: str):
    output_dir = f"{output_dir}/{model_name.replace("/", "_")}"
    os.makedirs(output_dir, exist_ok=True)
    # try:
    #     print(f"Directory '{output_dir}' created successfully.")
    # except FileExistsError:
    #     print(f"Directory '{output_dir}' created successfully.")

    print("--- Setting up Model and Tokenizer with Unsloth ---")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )

    if model_name != "unsloth/Qwen3-14B":
        _ = FastLanguageModel.for_inference(model)

    layers = model.model.layers
        
    num_layers = len(layers)
    all_layer_outputs = [torch.tensor(0) for _ in range(num_layers)]
    attachHooks(model, layers, all_layer_outputs)

    with open('task_suite.json', 'r') as f:
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
            input_ids=torch.tensor([input.tokenized]).to("cuda:0"),
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
    saveTasks(hidden_states_by_task, model_name, output_dir)
    graphByTask(tasks, hidden_states_by_task, num_layers, model_name, output_dir)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="VisualizeHiddenStates")
    parser.add_argument(
        "--model_name", type=str, choices=["llama-3.1-8b-instruct", "phi-4", "qwen3"], default="llama-3.1-8b-instruct"
    )
    parser.add_argument("--output_dir", type=str, default="./output")

    args = parser.parse_args()

    match str(args.model_name).lower():
        case "qwen3":
            model_name = "unsloth/Qwen3-14B"
        case "phi-4":
            model_name = "unsloth/Phi-4"
        case _:
            model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    
    main(model_name, args.output_dir)
