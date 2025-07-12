import torch
from transformers import AutoTokenizer
from unsloth import FastLanguageModel # <<< UNLOTH CHANGE >>>
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import re
from functools import partial
import random

# --- PART 1: DATA GENERATION (No changes needed) ---

def to_base(n, base):
    if n == 0: return "0"
    digits = [];
    while n: digits.append(str(n % base)); n //= base
    return "".join(digits[::-1])

def generate_arithmetic_data(base, num_digits, num_samples):
    data = []
    max_val = base**num_digits
    for _ in range(num_samples):
        n1, n2 = random.randint(0, max_val - 1), random.randint(0, max_val - 1)
        s = n1 + n2
        n1_base, n2_base, s_base = to_base(n1, base), to_base(n2, base), to_base(s, base)
        problem = f"{n1_base}+{n2_base}"
        cot_steps = [f"Step 1: The problem is {problem}.", f"Step 2: The answer is {s_base}."] # Simplified CoT for example
        cot_trace = "\n".join(cot_steps)
        data.append({"problem": problem, "answer": s_base, "cot": cot_trace})
    return data

# --- PART 2: GEOMETRIC ANALYSIS MODULE (No changes needed, it's compatible) ---

class GeometricAnalyzer:
    def __init__(self, model, tokenizer):
        self.model, self.tokenizer = model, tokenizer
        self.outputs, self.hook_handles = {}, []

    def _get_hook(self, layer_idx):
        def hook(module, input, output): self.outputs[layer_idx] = output[0].detach().cpu()
        return hook

    def analyze(self, instruction, examples, layers_to_probe):
        self.outputs.clear()
        prompt_parts, separator = [instruction] + examples, "\n\n"
        full_prompt_text = separator.join(prompt_parts)
        inputs = self.tokenizer(full_prompt_text, return_tensors="pt").to(self.model.device)
        end_indices, current_offset = [], 0
        for part in prompt_parts:
            part_tokens = self.tokenizer(part, add_special_tokens=False)['input_ids']
            end_indices.append(current_offset + len(part_tokens) - 1)
            separator_tokens = self.tokenizer(separator, add_special_tokens=False)['input_ids']
            current_offset += len(part_tokens) + len(separator_tokens)
        
        for i in layers_to_probe:
            handle = self.model.model.layers[i].register_forward_hook(self._get_hook(i))
            self.hook_handles.append(handle)
        
        with torch.no_grad(): self.model(**inputs)
        
        for handle in self.hook_handles: handle.remove()
        self.hook_handles.clear()

        results = {}
        for i in layers_to_probe:
            hidden_states = self.outputs[i]
            task_vec = hidden_states[0, end_indices[0], :]
            example_vecs = hidden_states[0, end_indices[1:], :]
            if len(example_vecs) > 1:
                rho_d = 1 / (torch.mean(torch.pdist(example_vecs)) + 1e-6)
            else: rho_d = 0.0
            centroid = torch.mean(example_vecs, dim=0)
            d_r = 1 - torch.nn.functional.cosine_similarity(task_vec, centroid, dim=0)
            results[i] = {"rho_d": rho_d.item(), "d_r": d_r.item()}
        return results

# --- PART 3: EVALUATION MODULE (No changes needed, it's compatible) ---

def parse_model_output(output_text):
    match = re.search(r'(\d+)\s*$', output_text.strip())
    return match.group(1) if match else ""

def evaluate_accuracy(model, tokenizer, test_set, anchor_prompt=""):
    correct, total = 0, len(test_set)
    for item in tqdm(test_set, desc="Evaluating Accuracy", leave=False):
        prompt = anchor_prompt + f"Problem: {item['problem']}\nAnswer: "
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20, pad_token_id=tokenizer.eos_token_id, use_cache=True)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]
        if parse_model_output(response) == item['answer']: correct += 1
    return correct / total if total > 0 else 0

# --- PART 4: MAIN EXPERIMENT ORCHESTRATION (with Unsloth changes) ---

import torch
from transformers import AutoTokenizer
from unsloth import FastLanguageModel
from peft import LoraConfig
from datasets import Dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import re
from functools import partial
import random
import gc

# --- (PART 1, 2, 3: DATA GENERATION, GEOMETRIC ANALYZER, EVALUATION are unchanged) ---
# ... (Keep the functions to_base, generate_arithmetic_data, GeometricAnalyzer, 
#      parse_model_output, and evaluate_accuracy exactly as they were) ...

# --- PART 4: MAIN EXPERIMENT ORCHESTRATION (HEAVILY MODIFIED FOR ROBUSTNESS) ---

def main():
    # --- EXPERIMENT CONFIGURATION ---
    NUM_RUNS = 5  # <<< Number of times to repeat the experiment
    ICL_K_VALUES = [0, 1, 2, 4, 8, 16] # k-shots for ICL evaluation
    GEOMETRIC_ANALYSIS_K = 8 # Number of examples for geometric analysis
    ICL_EVAL_SUBSET_SIZE = 30 # Use a subset of test data for faster ICL eval
    
    print("--- Setting up Model and Tokenizer with Unsloth ---")
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=2048,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )

    # --- Data Generation ---
    print("\n--- Generating Datasets ---")
    bases = [8, 9, 10]
    datasets = {}
    for base in bases:
        print(f"Generating data for Base {base}...")
        datasets[base] = {
            "train_2d": generate_arithmetic_data(base, 2, 1000),
            "test_2d_id": generate_arithmetic_data(base, 2, 250),
        }

    # --- Data Structures to store results across all runs ---
    all_runs_geometric_results = {base: [] for base in bases}
    all_runs_icl_results = {base: {"standard": [], "cot": []} for base in bases}
    
    # --- MAIN EXPERIMENTAL LOOP ---
    for run_idx in range(NUM_RUNS):
        print(f"\n{'='*20} STARTING RUN {run_idx + 1}/{NUM_RUNS} {'='*20}")
        gc.collect(); torch.cuda.empty_cache() # Clear memory between runs
        
        # --- Experiment 1: Geometric Analysis ---
        print("\n--- Experiment 1: Geometric Analysis ---")
        analyzer = GeometricAnalyzer(model, tokenizer)
        num_layers = model.config.num_hidden_layers
        layers_to_probe = list(range(num_layers))

        for base in bases:
            print(f"  Analyzing geometry for Base {base} (Run {run_idx + 1})...")
            instruction = f"Perform addition in base {base}."
            # <<< MODIFICATION: Randomly sample examples for this run >>>
            sampled_examples = random.sample(datasets[base]['train_2d'], GEOMETRIC_ANALYSIS_K)
            examples = [f"{item['problem']} -> {item['answer']}" for item in sampled_examples]
            
            run_geometry = analyzer.analyze(instruction, examples, layers_to_probe)
            all_runs_geometric_results[base].append(run_geometry)

        # --- Experiment 2: In-Context Learning Performance ---
        print("\n--- Experiment 2: In-Context Learning (ICL) ---")
        for base in bases:
            for use_cot in [False, True]:
                cot_str = "CoT" if use_cot else "Standard"
                print(f"  Running ICL for Base {base} ({cot_str}, Run {run_idx + 1})...")
                run_accuracies = []
                for k in ICL_K_VALUES:
                    anchor_prompt = ""
                    if k > 0:
                        # <<< MODIFICATION: Randomly sample examples for this ICL evaluation >>>
                        icl_examples = random.sample(datasets[base]['train_2d'], k)
                        prompt_parts = []
                        for item in icl_examples:
                            if use_cot: prompt_parts.append(f"Problem: {item['problem']}\nAnswer:\n{item['cot']}")
                            else: prompt_parts.append(f"Problem: {item['problem']}\nAnswer: {item['answer']}")
                        anchor_prompt = "\n\n".join(prompt_parts) + "\n\n"
                    
                    test_subset = random.sample(datasets[base]['test_2d_id'], ICL_EVAL_SUBSET_SIZE)
                    accuracy = evaluate_accuracy(model, tokenizer, test_subset, anchor_prompt)
                    run_accuracies.append(accuracy)
                
                result_key = "cot" if use_cot else "standard"
                all_runs_icl_results[base][result_key].append(run_accuracies)

    # --- AGGREGATION AND PLOTTING ---
    print(f"\n{'='*20} ALL {NUM_RUNS} RUNS COMPLETE. Aggregating and plotting results... {'='*20}")

    # --- Plot 1: Aggregated Geometric Analysis ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for base in bases:
        # Aggregate geometric data
        base_rho_d = np.array([[res[i]['rho_d'] for i in layers_to_probe] for res in all_runs_geometric_results[base]])
        base_d_r = np.array([[res[i]['d_r'] for i in layers_to_probe] for res in all_runs_geometric_results[base]])
        
        mean_rho_d, std_rho_d = np.mean(base_rho_d, axis=0), np.std(base_rho_d, axis=0)
        mean_d_r, std_d_r = np.mean(base_d_r, axis=0), np.std(base_d_r, axis=0)

        ax1.plot(layers_to_probe, mean_rho_d, marker='o', markersize=4, label=f'Base {base} Mean')
        ax1.fill_between(layers_to_probe, mean_rho_d - std_rho_d, mean_rho_d + std_rho_d, alpha=0.2)
        ax2.plot(layers_to_probe, mean_d_r, marker='o', markersize=4, label=f'Base {base} Mean')
        ax2.fill_between(layers_to_probe, mean_d_r - std_d_r, mean_d_r + std_d_r, alpha=0.2)
    
    ax1.set_title(f'Aggregated Pattern Density ($\rho_d$) vs. Layer ({NUM_RUNS} Runs)')
    ax1.set_ylabel('Density'); ax1.grid(True); ax1.legend()
    ax2.set_title(f'Aggregated Representational Mismatch ($d_r$) vs. Layer ({NUM_RUNS} Runs)')
    ax2.set_ylabel('Mismatch'); ax2.set_xlabel('Layer Index'); ax2.grid(True); ax2.legend()
    plt.tight_layout(); plt.show()

    # --- Plot 2: Aggregated ICL Performance ---
    plt.figure(figsize=(10, 6))
    for base in bases:
        # Aggregate Standard ICL
        standard_acc = np.array(all_runs_icl_results[base]['standard'])
        mean_standard, std_standard = np.mean(standard_acc, axis=0), np.std(standard_acc, axis=0)
        p = plt.plot(ICL_K_VALUES, mean_standard, marker='o', linestyle='-', label=f'Base {base} (Standard)')
        plt.fill_between(ICL_K_VALUES, mean_standard - std_standard, mean_standard + std_standard, alpha=0.2, color=p[0].get_color())
        
        # Aggregate CoT ICL
        cot_acc = np.array(all_runs_icl_results[base]['cot'])
        mean_cot, std_cot = np.mean(cot_acc, axis=0), np.std(cot_acc, axis=0)
        p = plt.plot(ICL_K_VALUES, mean_cot, marker='s', linestyle='--', label=f'Base {base} (CoT)')
        plt.fill_between(ICL_K_VALUES, mean_cot - std_cot, mean_cot + std_cot, alpha=0.2, color=p[0].get_color())

    plt.title(f'Aggregated ICL Accuracy vs. Number of Shots ({NUM_RUNS} Runs)')
    plt.xlabel('Number of Shots (k)'); plt.ylabel('Mean Accuracy'); 
    if max(ICL_K_VALUES) > 2: plt.xscale('log', base=2); 
    plt.xticks(ICL_K_VALUES, ICL_K_VALUES); plt.legend(); plt.grid(True); plt.show()

    # --- Plot 3: Aggregated Final Analysis ---
    print("\n--- Final Analysis: Linking Geometry to Performance ---")
    sweet_spot_layer = 28 # Choose a layer based on plots (e.g., near the bottom of the d_r valley)
    
    plt.figure(figsize=(8, 6))
    for base in bases:
        # Get the mean geometric metric from the sweet spot across all runs
        d_r_at_sweet_spot_all_runs = [run[sweet_spot_layer]['d_r'] for run in all_runs_geometric_results[base]]
        mean_mismatch = np.mean(d_r_at_sweet_spot_all_runs)
        
        # Get the mean performance metric, e.g., accuracy at k=8
        k_target_idx = ICL_K_VALUES.index(8)
        k_target_acc_all_runs = np.array(all_runs_icl_results[base]['standard'])[:, k_target_idx]
        mean_accuracy = np.mean(k_target_acc_all_runs)
        
        plt.scatter(mean_mismatch, mean_accuracy, s=200, label=f'Base {base}', alpha=0.8, zorder=3)

    plt.title('Mean Performance vs. Mean Initial Geometric Mismatch')
    plt.xlabel(f'Mean Mismatch $d_r$ @ Layer {sweet_spot_layer}'); plt.ylabel(f'Mean Standard ICL Accuracy @ k=8')
    plt.grid(True); plt.legend(); plt.show()

if __name__ == "__main__":
    main()