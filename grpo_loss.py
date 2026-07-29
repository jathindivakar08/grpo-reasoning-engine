import torch
import torch.nn as F

def get_per_token_log_probs(model, input_ids: torch.Tensor, logits_to_keep: int) -> torch.Tensor:
    """
    Computes log probabilities for the generated completion tokens.
    
    Args:
        model: The policy or reference language model.
        input_ids: Full sequence tensor containing [prompt_ids + completion_ids] shape (G, seq_len).
        logits_to_keep: The length of completion tokens to slice.
        
    Returns:
        per_token_log_ps: Log-probabilities tensor of shape (G, logits_to_keep).
    """
    # Run forward pass to get logits across sequence
    outputs = model(input_ids)
    logits = outputs.logits # Shape: (G, seq_len, vocab_size)

    # Shift logits and labels by 1 position for Causal LM next-token prediction
    # We slice logits corresponding only to the completion tokens
    logits = logits[:, -(logits_to_keep + 1):-1, :] # Shape: (G, logits_to_keep, vocab_size)
    target_ids = input_ids[:, -logits_to_keep:]    # Shape: (G, logits_to_keep)

    # Convert raw logits to log probabilities using log_softmax
    log_probs = torch.log_softmax(logits, dim=-1)

    # Gather log-probs corresponding to actual generated target token IDs
    per_token_log_ps = torch.gather(log_probs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    
    return per_token_log_ps


def compute_grpo_loss(
    policy_log_ps: torch.Tensor,
    old_policy_log_ps: torch.Tensor,
    ref_log_ps: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    epsilon: float = 0.2,
    beta: float = 0.04
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the GRPO Clipped Surrogate Loss and Per-Token KL Divergence Penalty.
    
    Equations:
        1. Importance Ratio: r_{i,t} = exp(log \pi_\theta(y_{i,t}) - log \pi_{old}(y_{i,t}))
        2. Clipped Surrogate: min(r_{i,t} * A_i, clip(r_{i,t}, 1-eps, 1+eps) * A_i)
        3. Unbiased KL Penalty: D_KL = exp(log \pi_{ref} - log \pi_\theta) - (log \pi_{ref} - log \pi_\theta) - 1
    """
    # 1. Expand advantages to match completion sequence length (G, 1) -> (G, seq_len)
    advantages_expanded = advantages.unsqueeze(-1).expand_as(policy_log_ps)

    # 2. Compute Probability Ratio r_{i,t}
    log_ratio = policy_log_ps - old_policy_log_ps
    ratio = torch.exp(log_ratio)

    # 3. Compute Clipped Surrogate Objective
    surr1 = ratio * advantages_expanded
    surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages_expanded
    policy_loss_per_token = -torch.min(surr1, surr2)

    # 4. Compute Unbiased Per-Token KL Divergence Penalty (Schulman Estimator)
    # D_KL = exp(log \pi_ref - log \pi_\theta) - (log \pi_ref - log \pi_\theta) - 1
    kl_log_ratio = ref_log_ps - policy_log_ps
    kl_per_token = torch.exp(kl_log_ratio) - kl_log_ratio - 1.0

    # 5. Combined GRPO Total Loss per token
    total_loss_per_token = policy_loss_per_token + (beta * kl_per_token)

    # 6. Apply Completion Attention Mask (ignore padding tokens)
    masked_loss = (total_loss_per_token * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
    masked_policy_loss = (policy_loss_per_token * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
    masked_kl = (kl_per_token * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)

    return masked_loss, masked_policy_loss, masked_kl


if __name__ == "__main__":
    print("⚡ Testing Custom GRPO Loss Module...\n")

    # Mock Tensors: Batch G = 4, Completion Sequence Length T = 8
    G, T = 4, 8
    
    # Synthetic log probabilities
    mock_policy_log_ps = torch.randn(G, T, requires_grad=True)
    mock_old_log_ps = mock_policy_log_ps.detach() # Ratio starts at 1.0
    mock_ref_log_ps = mock_policy_log_ps.detach() + 0.05 # Slight drift
    
    # Mock advantages from Phase 2
    mock_advantages = torch.tensor([1.0, -1.0, -1.0, 1.0])
    
    # Completion mask (all 1s, no padding)
    mock_mask = torch.ones(G, T)

    loss, policy_loss, kl_penalty = compute_grpo_loss(
        mock_policy_log_ps,
        mock_old_log_ps,
        mock_ref_log_ps,
        mock_advantages,
        mock_mask
    )

    print(f"📊 Calculated GRPO Total Loss: {loss.item():.6f}")
    print(f"   ├─ Policy Objective Loss:  {policy_loss.item():.6f}")
    print(f"   └─ KL Divergence Penalty:  {kl_penalty.item():.6f}")

    # Backpropagation Check
    loss.backward()
    print(f"\n✅ Backward pass executed cleanly! Gradient shape: {mock_policy_log_ps.grad.shape}")