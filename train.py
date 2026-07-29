import json
import os
import torch
from torch.optim import AdamW
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from verifier import compute_reward
from sampler import generate_group_samples, compute_group_advantages
from grpo_loss import get_per_token_log_probs, compute_grpo_loss

# --- Hyperparameters ---
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
GROUP_SIZE = 4            # Number of candidate completions per prompt (G)
MAX_NEW_TOKENS = 256      # Max generation length
LEARNING_RATE = 5e-6      # Stable learning rate for policy gradient
BETA = 0.04               # KL penalty coefficient
EPSILON = 0.2             # PPO clipping ratio
NUM_STEPS = 50            # Set to 50 steps (or change to 300 for longer training)
SAVE_EVERY = 10           # Save model checkpoint every N steps
METRICS_FILE = "metrics.json"

def setup_training_environment():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"🖥️ Using Hardware Backend: {device.upper()}")
    print(f"📥 Loading Tokenizer & Base Model: {MODEL_NAME}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16
    ).to(device)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    return device, tokenizer, model, optimizer


def train_grpo():
    device, tokenizer, model, optimizer = setup_training_environment()

    print("\n📚 Loading GSM8K Math Training Dataset...")
    dataset = load_dataset("openai/gsm8k", "main", split="train")

    print(f"🚀 Starting GRPO Training Loop ({NUM_STEPS} Steps)...\n")
    print("Step | Total Loss | Policy Loss | KL Penalty | Avg Reward | Accuracy | Avg Token Len")
    print("-" * 80)

    history = []

    for step in range(1, NUM_STEPS + 1):
        sample = dataset[(step - 1) % len(dataset)]
        question = sample["question"]
        ground_truth = sample["answer"].split("####")[-1].strip()

        # 1. Sample G completions
        model.eval()
        prompt_ids, completion_ids, completions = generate_group_samples(
            model, tokenizer, question,
            group_size=GROUP_SIZE,
            max_new_tokens=MAX_NEW_TOKENS,
            device=device
        )

        # 2. Rewards & Group Advantages
        rewards, advantages = compute_group_advantages(completions, ground_truth)
        advantages = advantages.to(device)

        full_input_ids = torch.cat([prompt_ids.repeat(GROUP_SIZE, 1), completion_ids], dim=1)
        completion_mask = (completion_ids != tokenizer.pad_token_id).float().to(device)
        completion_len = completion_ids.shape[1]

        # 3. Policy Log-Probs
        model.train()
        policy_log_ps = get_per_token_log_probs(model, full_input_ids, completion_len)
        old_policy_log_ps = policy_log_ps.detach()

        # 4. Reference Model Log-Probs (Disabled LoRA)
        with torch.no_grad():
            with model.disable_adapter():
                ref_log_ps = get_per_token_log_probs(model, full_input_ids, completion_len)

        # 5. GRPO Loss & KL Penalty
        loss, policy_loss, kl_penalty = compute_grpo_loss(
            policy_log_ps=policy_log_ps,
            old_policy_log_ps=old_policy_log_ps,
            ref_log_ps=ref_log_ps,
            advantages=advantages,
            completion_mask=completion_mask,
            epsilon=EPSILON,
            beta=BETA
        )

        # 6. Optimization Step
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 7. Operational Metrics
        avg_reward = rewards.mean().item()
        accuracy = (rewards == 1.0).float().mean().item() * 100.0
        avg_tokens = completion_mask.sum(dim=1).mean().item()

        step_data = {
            "step": step,
            "total_loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "kl_penalty": kl_penalty.item(),
            "avg_reward": avg_reward,
            "accuracy": accuracy,
            "avg_token_len": avg_tokens
        }
        history.append(step_data)

        print(f"{step:02d}/{NUM_STEPS} | {loss.item():10.4f} | {policy_loss.item():11.4f} | {kl_penalty.item():10.4f} | {avg_reward:10.2f} | {accuracy:7.1f}% | {avg_tokens:13.1f}")

        # Update metrics.json on every step
        with open(METRICS_FILE, "w") as f:
            json.dump(history, f, indent=2)

        # Periodic checkpointing
        if step % SAVE_EVERY == 0 or step == NUM_STEPS:
            ckpt_dir = f"./checkpoints/step_{step}"
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)

    # Also save to main output directory
    model.save_pretrained("./grpo_qwen_lora")
    tokenizer.save_pretrained("./grpo_qwen_lora")
    print(f"\n✅ Training Complete! Saved `metrics.json`, checkpoints, and `./grpo_qwen_lora`.")

if __name__ == "__main__":
    train_grpo()