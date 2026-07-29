import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sampler import format_math_prompt
from verifier import extract_boxed_answer

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_PATH = "./grpo_qwen_lora"

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

print("📥 Loading Base Model and Trained GRPO LoRA Adapters...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16).to(device)
model = PeftModel.from_pretrained(base_model, LORA_PATH).to(device)

# Unseen test question
test_question = "Janet buys 3 bags of chips for $2 each and 2 sodas for $1.50 each. How much change does she get back from a $10 bill?"
prompt = format_math_prompt(test_question)

inputs = tokenizer(prompt, return_tensors="pt").to(device)

print(f"\n❓ Question: {test_question}\n")
print("🤖 Generating Reasoning Chain...")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7)

completion_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
boxed = extract_boxed_answer(completion_text)

print("--- Trained Model Response ---")
print(completion_text)
print("-" * 40)
print(f"🎯 Extracted Final Answer: \\boxed{{{boxed}}}")