import torch
from common.types import Task, Input, Range


def attachHooks(model, layers, all_layer_outputs):
    num_layers = len(layers)

    def get_layer_output_hook(layer_idx):
        def hook(module, input, output):
            hidden_states = output[0]
            if hidden_states.shape[1] > 1:
                all_layer_outputs[layer_idx] = hidden_states.detach()

        return hook

    hook_handles = []
    for i in range(num_layers):
        target_layer = layers[i]
        handle = target_layer.register_forward_hook(get_layer_output_hook(i))
        hook_handles.append(handle)
    print(f"Attached {len(hook_handles)} hooks to layers 0 through {num_layers - 1}.")


def formatChoices(choices: dict[str, list[str]]):
    return "\n".join([f"{l}) {t}" for l, t in zip(choices["label"], choices["text"])])


def formatTasks(
    model_name: str, suite: list, category_examples: dict[str, str]
) -> list[Task]:
    tasks = []

    for t in suite:
        instruction = t["instruction"]
        if "choices" in t:
            instruction += "\n" + formatChoices(t["choices"])

        examples = category_examples[t["category"]]
        print(examples)

        match model_name:
            case "unsloth/gemma-3-4b-it":
                tasks.append(
                    Task(
                        category=str(t["category"]),
                        input_header="<bos><start_of_turn>user\nAnswer the question using Answer: ... Below I've provided a few examples.\n\n",
                        examples="Examples:\n" + examples + "\n\n",
                        task="Question:\n" + instruction,
                        input_footer="<end_of_turn>\n<start_of_turn>model\n",
                    ),
                )
            case "unsloth/Phi-4":
                tasks.append(
                    Task(
                        category=str(t["category"]),
                        input_header="<|im_start|>user<|im_sep|>\nAnswer the question using Answer: ... Below I've provided a few examples.\n\n",
                        examples="Examples:\n" + examples + "\n\n",
                        task="Question:\n" + instruction,
                        input_footer="\n\n<|im_end|><|im_start|>assistant<|im_sep|>\n\n",
                    ),
                )
            case "meta-llama/Meta-Llama-3.1-8B-Instruct":
                tasks.append(
                    Task(
                        category=str(t["category"]),
                        input_header="<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 July 2024\n\nToday Date: 26 July 2024\n\n<|eot_id|><|start_header_id|>human<|end_header_id|>\nAnswer the question using Answer: ... Below I've provided a few examples.\n\n",
                        examples="Examples:\n" + examples + "\n\n",
                        task="Question:\n" + instruction,
                        input_footer="\n\n<|start_header_id|>assistant<|end_header_id|>\n\n",
                    ),
                )
    return tasks


def processTaskInput(task: Task, tokenizer) -> Input:
    """
    Tokenizes and processes a task input to feed into a model. Records the indices of where the task and example tokens are.

    Args:
        task (Task): task to feed to the model
        tokenizer: model tokenizer

    Returns:
        Input: a Named tuple that stores the task indices, example indices, and tokenized input.
    """
    lengths = []
    recon_tokens = []
    # print(task)
    for i, x in enumerate(task[1:]):
        section_tokens = tokenizer.encode(x)
        recon_tokens.extend(section_tokens)
        lengths.append(len(section_tokens))

    indices = []
    lower = 0
    for length in lengths:
        indices.append(Range(lower, length + lower))
        lower += length

    return Input(
        task_indices=indices[2],
        examples_indices=indices[1],
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


# def calcHiddenStates(
#     processedTasks: list[tuple[Task, Input]],
#     num_layers: int,
#     task_layer_outputs: dict[int, list[torch.Tensor]],
# ) -> dict[tuple[int, str], HiddenStatesByLayer]:
#     hidden_states_by_task: dict[tuple[int, str], HiddenStatesByLayer] = {}
    
#     for task_num, (task, input) in enumerate(processedTasks):
#         rho_d_by_layer: list[float] = []
#         d_r_by_layer: list[float] = []
#         task_vecs_by_layer: list[torch.Tensor] = []
#         example_vecs_by_layer: list[torch.Tensor] = []

#         for i in range(num_layers):
#             layer_hidden_states = task_layer_outputs[task_num][i]

#             lower, upper = input.task_indices
#             task_vecs = layer_hidden_states[0, lower:upper, :]
#             lower, upper = input.examples_indices
#             example_vecs = layer_hidden_states[0, lower:upper, :]
#             task_vecs_by_layer.append(task_vecs)
#             example_vecs_by_layer.append(example_vecs)

#             if len(example_vecs) > 1:
#                 mean_cos_dist = pairwise_cosine_distance(example_vecs)
#                 rho_d = 1 / (mean_cos_dist + 1e-6)
#             else:
#                 print(
#                     "WARNING: example token size 1. If this is what you want, ignore this message."
#                 )
#                 rho_d = torch.tensor(0)
#             rho_d_by_layer.append(rho_d.item())

#             pattern_centroid = torch.mean(example_vecs, dim=0)
#             task_centroid = torch.mean(task_vecs, dim=0)
#             d_r = 1 - torch.nn.functional.cosine_similarity(
#                 task_centroid, pattern_centroid, dim=0
#             )
#             d_r_by_layer.append(d_r.item())

#         hidden_states_by_task[(task_num, task.category)] = HiddenStatesByLayer(
#             rho_d_by_layer, d_r_by_layer, task_vecs_by_layer, example_vecs_by_layer
#         )
#     return hidden_states_by_task