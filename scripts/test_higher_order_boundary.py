import unittest

import torch
import tinycudann as tcnn


def hash_grid():
    return tcnn.Encoding(3, {
        "otype": "HashGrid", "n_levels": 2, "n_features_per_level": 2,
        "log2_hashmap_size": 8, "base_resolution": 4,
        "per_level_scale": 2.0, "interpolation": "Linear",
    })


class HigherOrderBoundaryTest(unittest.TestCase):
    def test_input_gradient_training_still_runs(self):
        for jit in (False, True):
            with self.subTest(jit=jit):
                model = hash_grid()
                model.jit_fusion = jit
                x = torch.full((128, 3), 0.37, device="cuda").requires_grad_()
                y = model(x)
                g = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
                loss = g.square().mean()
                grad = torch.autograd.grad(loss, model.params)[0]
                self.assertTrue(bool(torch.isfinite(grad).all()))
                self.assertGreater(float(grad.norm()), 0)
                with torch.no_grad():
                    model.params -= 1e-3 * grad

    def test_parameter_gradient_second_order_is_explicitly_unsupported(self):
        for jit in (False, True):
            with self.subTest(jit=jit):
                model = hash_grid()
                model.jit_fusion = jit
                x = torch.tensor([[0.33, 0.44, 0.55]], device="cuda").repeat(128, 1).requires_grad_()
                y = model(x)
                v = torch.ones_like(y)
                param_grad = torch.autograd.grad(
                    y, model.params, grad_outputs=v, create_graph=True
                )[0]
                score = param_grad.float().square().sum()
                with self.assertRaisesRegex(NotImplementedError, "parameter gradients"):
                    torch.autograd.grad(score, x)

    def test_unsupported_direction_is_not_mathematically_zero(self):
        model = hash_grid()
        model.jit_fusion = False
        x = torch.tensor([[0.33, 0.44, 0.55]], device="cuda").repeat(128, 1)
        v = torch.ones((128, model.n_output_dims), device="cuda", dtype=model.dtype)
        direction = torch.zeros_like(x)
        direction[:, 0] = 1

        def score_at(x_value):
            y = model(x_value)
            grad = torch.autograd.grad(y, model.params, grad_outputs=v)[0]
            return float(grad.float().square().sum())

        epsilon = 0.003
        finite_difference = (
            score_at((x + epsilon * direction).contiguous())
            - score_at((x - epsilon * direction).contiguous())
        ) / (2 * epsilon)
        self.assertGreater(abs(finite_difference), 1000)


if __name__ == "__main__":
    unittest.main()
