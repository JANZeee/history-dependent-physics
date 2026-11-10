import unittest
import torch
from data import make_dataset
from models import make_model


class ModelInvariants(unittest.TestCase):
    def test_shapes_causality_and_gradients(self):
        torch.set_num_threads(2)
        for task in ("thermal", "fatigue"):
            d = make_dataset(task, 4, 8, 93001)
            x, init = torch.from_numpy(d["x"]), torch.from_numpy(d["initial"])
            for name in ("gru", "constrained"):
                torch.manual_seed(91)
                m = make_model(name, task)
                y = m(x, init)
                self.assertEqual(y.shape, (4,8,1))
                altered = x.clone()
                altered[:,4:] = 0
                torch.testing.assert_close(y[:,:4], m(altered, init)[:,:4])
                y.square().mean().backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_constrained_passive_behavior(self):
        for task in ("thermal", "fatigue"):
            d = make_dataset(task, 4, 16, 93002, mode="passive_tail", active_length=8)
            x, init = torch.from_numpy(d["x"]), torch.from_numpy(d["initial"])
            m = make_model("constrained", task)
            y = m(x, init).detach()[...,0]
            if task == "thermal":
                self.assertTrue(torch.all(y >= 0))
                self.assertTrue(torch.all(y[:,8:] <= y[:,7:-1]+1e-7))
            else:
                self.assertTrue(torch.all(y[:,1:] >= y[:,:-1]))
                torch.testing.assert_close(y[:,8:], y[:,7:8].expand(-1,8))
                failed = m(x, torch.ones_like(init))
                torch.testing.assert_close(failed, torch.ones_like(failed))


if __name__ == "__main__":
    unittest.main()
