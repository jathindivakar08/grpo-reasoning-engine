import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from sampler import format_math_prompt
from verifier import compute_reward, extract_boxed_answer

BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
CHECKPOINT_PATH = "./grpo_qwen_lora"


device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

print("📥 Loading Base Model and Trained GRPO Model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME, torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16
).to(device)

# Load GRPO model with LoRA adapters
grpo_model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH).to(device)

# Load test dataset
test_dataset = load_dataset("openai/gsm8k", "main", split="test")

print("\n🧪 Running Side-by-Side Comparison on 5 Unseen Test Problems...\n")

for i in range(5):
    sample = test_dataset[i]
    question = sample["question"]
    ground_truth = sample["answer"].split("####")[-1].strip()

    prompt = format_math_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # 1. Base Model Response (LoRA Disabled)
    with torch.no_grad():
        with grpo_model.disable_adapter():
            base_out = grpo_model.generate(**inputs, max_new_tokens=256, do_sample=False)
    
    base_text = tokenizer.decode(base_out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    base_boxed = extract_boxed_answer(base_text)
    base_reward = compute_reward(base_text, ground_truth)

    # 2. GRPO Trained Model Response
    with torch.no_grad():
        grpo_out = grpo_model.generate(**inputs, max_new_tokens=256, do_sample=False)

    grpo_text = tokenizer.decode(grpo_out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    grpo_boxed = extract_boxed_answer(grpo_text)
    grpo_reward = compute_reward(grpo_text, ground_truth)

    print(f"============================ Test Problem {i+1} ============================")
    print(f"❓ Question: {question}")
    print(f"🎯 Ground Truth: {ground_truth}\n")
    print(f"🔴 Base Model Answer:  \\boxed{{{base_boxed}}} | Correct: {'✅' if base_reward == 1.0 else '❌'}")
    print(f"🟢 GRPO Model Answer:  \\boxed{{{grpo_boxed}}} | Correct: {'✅' if grpo_reward == 1.0 else '❌'}\n")

print("✅ Side-by-side benchmark comparison complete!")