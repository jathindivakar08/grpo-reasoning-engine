import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from verifier import compute_reward

def format_math_prompt(question: str) -> str:
    """
    Formats the raw math problem using the standard reasoning instruction template.
    Enforces step-by-step CoT reasoning and \boxed{} answer placement.
    """
    system_prompt = (
        "Please reason step by step, and put your final answer within \\boxed{}."
    )
    return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"


def generate_group_samples(
    model,
    tokenizer,
    prompt_text: str,
    group_size: int = 4,
    max_new_tokens: int = 256,
    temperature: float = 0.9,
    device: str = "cpu"
):
    """
    Generates G completions for a single prompt using temperature sampling.
    Returns:
        - prompt_ids: Tensor of shape (1, prompt_len)
        - completion_ids: Tensor of shape (G, max_completion_len)
        - completion_texts: List of G decoded string completions
    """
    formatted_prompt = format_math_prompt(prompt_text)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    prompt_ids = inputs["input_ids"]
    prompt_len = prompt_ids.shape[1]

    # Expand prompt tensor to batch size G for parallel generation
    expanded_input_ids = prompt_ids.repeat(group_size, 1)
    attention_mask = inputs["attention_mask"].repeat(group_size, 1)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=expanded_input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Slice out prompt tokens to keep only the generated completion tokens
    completion_ids = output_ids[:, prompt_len:]

    # Decode each completion sequence to text
    completion_texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

    return prompt_ids, completion_ids, completion_texts


def compute_group_advantages(
    completion_texts: list[str],
    ground_truth: str,
    eps: float = 1e-8
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates scalar rewards and normalizes relative advantages across group G.
    Formula: A_i = (r_i - mean(r)) / (std(r) + eps)
    """
    rewards_list = [compute_reward(text, ground_truth) for text in completion_texts]
    rewards = torch.tensor(rewards_list, dtype=torch.float32)

    mean_r = rewards.mean()
    std_r = rewards.std(unbiased=False)

    # Standardize rewards across the group to produce zero-mean advantage signals
    if std_r < eps:
        advantages = torch.zeros_like(rewards)
    else:
        advantages = (rewards - mean_r) / (std_r + eps)

    return rewards, advantages


if __name__ == "__main__":
    print("⚡ Testing Group Sampler & Advantage Logic...\n")

    # Select hardware backend (CUDA for Nvidia GPU, MPS for Apple Silicon, CPU as fallback)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"🖥️ Using Device: {device.upper()}")

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"📥 Loading base model: {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16
    ).to(device)

    # Test Sample Math Problem from GSM8K
    test_question = "Natalia sold cookies to 48 of her friends. If 1/2 of her friends bought chocolate chip cookies, how many friends bought chocolate chip cookies?"
    test_ground_truth = "24"

    print(f"\n❓ Prompt Question: {test_question}")
    print(f"🎯 Ground Truth: {test_ground_truth}\n")

    G = 4
    prompt_ids, completion_ids, completions = generate_group_samples(
        model, tokenizer, test_question, group_size=G, device=device
    )

    rewards, advantages = compute_group_advantages(completions, test_ground_truth)

    print("--- Group Completions & GRPO Advantage Scores ---")
    for i in range(G):
        print(f"\n[Sample {i+1}]")
        print(f"Text: {completions[i][:120]}...")
        print(f"Reward r_{i}: {rewards[i].item():.1f} | Advantage A_{i}: {advantages[i].item():.4f}")

    print("\n✅ Group sampling and relative advantage calculations executed successfully!")