import json
import matplotlib.pyplot as plt

def plot_grpo_training():
    try:
        with open("metrics.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("❌ Error: metrics.json not found. Run train.py first!")
        return

    steps = [d["step"] for d in data]
    accuracy = [d["accuracy"] for d in data]
    avg_tokens = [d["avg_token_len"] for d in data]
    kl_penalty = [d["kl_penalty"] for d in data]

    # Calculate moving averages (smoothing window = 10)
    window = 10
    def moving_avg(arr):
        return [sum(arr[max(0, i-window):i+1])/len(arr[max(0, i-window):i+1]) for i in range(len(arr))]

    smooth_acc = moving_avg(accuracy)
    smooth_tokens = moving_avg(avg_tokens)

    plt.figure(figsize=(14, 5))

    # Graph 1: Reasoning Accuracy Curve
    plt.subplot(1, 2, 1)
    plt.plot(steps, accuracy, alpha=0.3, color="blue", label="Raw Accuracy")
    plt.plot(steps, smooth_acc, color="blue", linewidth=2, label="10-Step Moving Avg")
    plt.title("GSM8K Math Accuracy (%) vs. Training Steps", fontsize=12, fontweight="bold")
    plt.xlabel("Step")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    # Graph 2: Reasoning Token Expansion ("The Aha Moment")
    plt.subplot(1, 2, 2)
    plt.plot(steps, avg_tokens, alpha=0.3, color="green", label="Raw Token Count")
    plt.plot(steps, smooth_tokens, color="green", linewidth=2, label="10-Step Moving Avg")
    plt.title("Completion Reasoning Token Length vs. Training Steps", fontsize=12, fontweight="bold")
    plt.xlabel("Step")
    plt.ylabel("Avg Tokens Generated")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig("grpo_training_curves.png", dpi=300)
    print("📊 Training curves saved to `grpo_training_curves.png`!")
    plt.show()

if __name__ == "__main__":
    plot_grpo_training()