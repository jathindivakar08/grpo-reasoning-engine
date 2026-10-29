"""CPU regressions: masked completion positions must not affect loss or gradients."""

import math
import unittest
from contextlib import contextmanager
from types import SimpleNamespace

import torch

from grpo_loss import compute_grpo_loss, get_per_token_log_probs


class TinyAdapterModel(torch.nn.Module):
    """Finite causal logits with an adapter difference only at padding."""

    def __init__(self, dtype, gap):
        super().__init__()
        logits = torch.zeros(4, 3, 2, dtype=dtype)
        logits[:, 1, 0] = -gap
        self.adapter = torch.nn.Parameter(logits)
        self.adapter_enabled = True

    def forward(self, input_ids):
        logits = self.adapter if self.adapter_enabled else torch.zeros_like(self.adapter)
        return SimpleNamespace(logits=logits)

    @contextmanager
    def disable_adapter(self):
        self.adapter_enabled = False
        try:
            yield
        finally:
            self.adapter_enabled = True


class CompletionMaskTests(unittest.TestCase):
    def check_producer_padding(self, dtype, gap):
        model = TinyAdapterModel(dtype, gap)
        input_ids = torch.tensor([[0, 1, 0]]).repeat(4, 1)
        completion_ids = input_ids[:, -2:]
        mask = (completion_ids != 0).to(dtype)
        advantages = torch.tensor([1, 1, -1, -1], dtype=dtype)
        policy = get_per_token_log_probs(model, input_ids, 2)
        with torch.no_grad(), model.disable_adapter():
            reference = get_per_token_log_probs(model, input_ids, 2)

        # The actual producer returns valid log-probabilities even for padding.
        self.assertTrue(torch.isfinite(policy).all())
        self.assertTrue(torch.isfinite(reference).all())
        self.assertTrue((policy <= 0).all())
        outputs = compute_grpo_loss(policy, policy.detach(), reference, advantages, mask)
        for value in outputs:
            torch.testing.assert_close(value, value.new_zeros(()))
        outputs[0].backward()

        expected = torch.zeros_like(model.adapter)
        expected[:, 0, 0] = advantages / 8
        expected[:, 0, 1] = -advantages / 8
        torch.testing.assert_close(model.adapter.grad, expected)
        torch.optim.SGD(model.parameters(), lr=0.01).step()
        self.assertTrue(torch.isfinite(model.adapter).all())

    def test_real_logprob_producer_float16(self):
        self.check_producer_padding(torch.float16, 20)

    def test_real_logprob_producer_float32(self):
        self.check_producer_padding(torch.float32, 120)

    def test_real_logprob_producer_float64(self):
        self.check_producer_padding(torch.float64, 800)

    def test_masked_policy_ratio_overflow_is_ignored(self):
        policy = torch.tensor([[-1.0, -1.0]], requires_grad=True)
        old = torch.tensor([[-1.0, -120.0]])
        reference = policy.detach()
        outputs = compute_grpo_loss(
            policy, old, reference, torch.tensor([-1.0]), torch.tensor([[1.0, 0.0]])
        )
        torch.testing.assert_close(outputs[0], torch.tensor(1.0))
        outputs[0].backward()
        torch.testing.assert_close(policy.grad, torch.tensor([[1.0, 0.0]]))

    def test_all_masked_batch_has_zero_loss_and_gradients(self):
        policy = torch.tensor([[-120.0, -1.0]], requires_grad=True)
        outputs = compute_grpo_loss(
            policy, torch.tensor([[-1.0, -120.0]]), torch.tensor([[-1.0, -1.0]]),
            torch.tensor([1.0]), torch.zeros(1, 2)
        )
        for value in outputs:
            torch.testing.assert_close(value, torch.tensor(0.0))
        outputs[0].backward()
        torch.testing.assert_close(policy.grad, torch.zeros_like(policy))

    def test_zero_kl_coefficient_still_ignores_padding(self):
        policy = torch.tensor([[-1.0, -120.0]], requires_grad=True)
        outputs = compute_grpo_loss(
            policy, policy.detach(), torch.tensor([[-1.0, -1.0]]),
            torch.tensor([1.0]), torch.tensor([[1.0, 0.0]]), beta=0.0
        )
        torch.testing.assert_close(outputs[0], torch.tensor(-1.0))
        outputs[0].backward()
        torch.testing.assert_close(policy.grad, torch.tensor([[-1.0, 0.0]]))

    def test_normal_active_tokens_match_scalar_objective(self):
        policy = torch.tensor([[-1.0, -3.0], [-1.2, -4.0]], dtype=torch.float64,
                              requires_grad=True)
        old = torch.tensor([[-1.1, -3.0], [-1.0, -4.0]], dtype=torch.float64)
        reference = torch.tensor([[-0.9, -3.0], [-0.9, -4.0]], dtype=torch.float64)
        outputs = compute_grpo_loss(
            policy, old, reference, torch.tensor([2.0, -1.0], dtype=torch.float64),
            torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
        )
        ratio_a, ratio_b = math.exp(0.1), math.exp(-0.2)
        expected_policy = (-2 * ratio_a + ratio_b) / 2
        expected_kl = (math.expm1(0.1) - 0.1 + math.expm1(0.3) - 0.3) / 2
        expected = [expected_policy + 0.04 * expected_kl, expected_policy, expected_kl]
        for observed, value in zip(outputs, expected):
            self.assertAlmostEqual(observed.item(), value, places=12)
        outputs[0].backward()
        expected_gradient = policy.new_tensor([
            [(-2 * ratio_a + 0.04 * (1 - math.exp(0.1))) / 2, 0],
            [(ratio_b + 0.04 * (1 - math.exp(0.3))) / 2, 0],
        ])
        torch.testing.assert_close(policy.grad, expected_gradient)


if __name__ == '__main__':
    unittest.main()
