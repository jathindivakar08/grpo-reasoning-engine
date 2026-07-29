import re
import math

def extract_boxed_answer(text: str) -> str | None:
    """
    Extracts the content inside the last \boxed{...} tag from generated text.
    Handles nested curly braces if present.
    """
    if not text:
        return None

    # Find all occurrences of \boxed{
    matches = [m.start() for m in re.finditer(r'\\boxed\{', text)]
    if not matches:
        return None

    # Extract from the last \boxed{ occurrence to account for step-by-step scratchpads
    last_match_idx = matches[-1]
    start_idx = last_match_idx + len(r'\boxed{')
    
    brace_count = 1
    i = start_idx
    while i < len(text) and brace_count > 0:
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
        i += 1

    if brace_count == 0:
        return text[start_idx:i-1].strip()
    return None


def clean_answer_string(ans_str: str) -> str:
    """
    Normalizes LaTeX and numeric formatting for clean comparison.
    (e.g., "$42.0$" -> "42", "  1,000 " -> "1000")
    """
    if not ans_str:
        return ""
    
    # Remove leading/trailing whitespaces and currency/math symbols
    ans = ans_str.strip().replace("$", "").replace(",", "")
    
    # Try converting float to integer if equivalent (e.g. 42.0 -> 42)
    try:
        val = float(ans)
        if val.is_integer():
            return str(int(val))
        return str(val)
    except ValueError:
        pass

    return ans.lower()


def compute_reward(completion_text: str, ground_truth: str) -> float:
    """
    Deterministic Reward Evaluator:
    - Returns 1.0 if the extracted \boxed{...} answer matches ground_truth.
    - Returns 0.0 otherwise.
    """
    extracted = extract_boxed_answer(completion_text)
    if extracted is None:
        return 0.0

    clean_extracted = clean_answer_string(extracted)
    clean_gt = clean_answer_string(ground_truth)

    if clean_extracted == clean_gt:
        return 1.0
    
    return 0.0


if __name__ == "__main__":
    # --- Quick Unit Tests ---
    print("🧪 Running Verifier Engine Unit Tests...\n")

    sample_completion_1 = """
    To solve the equation 2x + 4 = 10:
    1. Subtract 4 from both sides: 2x = 6
    2. Divide by 2: x = 3
    The final answer is \\boxed{3}.
    """
    sample_gt_1 = "3"
    r1 = compute_reward(sample_completion_1, sample_gt_1)
    print(f"Test 1 (Exact Match): Reward = {r1} | Expected = 1.0")
    assert r1 == 1.0

    sample_completion_2 = """
    The total cost of the items is $1,250.00 dollars.
    \\boxed{1,250}
    """
    sample_gt_2 = "1250"
    r2 = compute_reward(sample_completion_2, sample_gt_2)
    print(f"Test 2 (Format Normalization): Reward = {r2} | Expected = 1.0")
    assert r2 == 1.0

    sample_completion_3 = """
    I think the answer is 42, but I forgot to wrap it in boxed tags.
    """
    sample_gt_3 = "42"
    r3 = compute_reward(sample_completion_3, sample_gt_3)
    print(f"Test 3 (Missing Boxed Tag): Reward = {r3} | Expected = 0.0")
    assert r3 == 0.0

    print("\n✅ All verifier unit tests passed successfully!")