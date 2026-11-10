import unittest

import numpy as np
import torch

from data import make_dataset
from models import GRU_NAMES, TASK_MODELS, make_model


class ModelsTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_identical_initialization_in_each_family(self):
        for task in TASK_MODELS:
            for family in (GRU_NAMES, tuple(n for n in TASK_MODELS[task] if n.startswith('state_'))):
                params = []
                for name in family:
                    torch.manual_seed(101)
                    params.append(torch.cat([p.detach().flatten() for p in make_model(name, task).parameters()]))
                for v in params[1:]:
                    self.assertTrue(torch.equal(v, params[0]))

    def test_causality_finite_gradients_and_projection(self):
        for task, names in TASK_MODELS.items():
            d = make_dataset(task, 8, 16, 91001)
            x, initial, y = [torch.from_numpy(d[k]) for k in ('x','initial','y')]
            for name in names:
                torch.manual_seed(101)
                m = make_model(name, task)
                p = m(x, initial)
                changed = x.clone(); changed[:, 8:, 0] = 0
                self.assertTrue(torch.equal(p[:, :8], m(changed, initial)[:, :8]), (task,name))
                (p-y).square().mean().backward()
                grads = torch.cat([v.grad.flatten() for v in m.parameters() if v.grad is not None])
                self.assertTrue(torch.isfinite(grads).all(), (task,name))
                self.assertGreater(grads.abs().sum().item(), 0, (task,name))
                if 'projected' in name:
                    prev = torch.cat((initial.unsqueeze(1), p[:, :-1]), 1)
                    rest = x[..., 0] == 0
                    if task == 'fatigue':
                        self.assertTrue((p >= prev).all())
                        self.assertTrue((p <= 1).all())
                        self.assertTrue(torch.equal(p[rest], prev[rest]))
                    else:
                        self.assertTrue((p >= 0).all())
                        self.assertTrue((p[rest] <= prev[rest]).all())

    def test_nested_updates_only_named_operator_changes(self):
        torch.manual_seed(101)
        m = make_model('state_free','fatigue')
        u = torch.tensor([[0.,1.,.8,.7,1.], [.4,1.2,.8,.7,1.]])
        a = torch.tensor([[.14],[.35]])
        free = m.step(u,a)-a
        m.name='state_gated'
        gated=m.step(u,a)-a
        torch.testing.assert_close(gated,free*u[:, :1],atol=2e-8,rtol=1e-4)
        m.name='state_positive'; positive=m.step(u,a)
        self.assertTrue((positive>=a).all())
        self.assertEqual(positive[0].item(),a[0].item())
        m.name='state_full'; full=m.step(u,a)
        torch.testing.assert_close(full,positive.clamp_max(1))

    def test_free_has_no_hidden_clamp_and_full_absorbs(self):
        m=make_model('state_free','fatigue')
        for p in m.parameters():
            p.data.zero_()
        m.net[-1].bias.data.fill_(-10)
        u=torch.tensor([[0.,1.,.8,.7,1.]])
        a=torch.tensor([[.08]])
        self.assertLess(m.step(u,a).item(),0)
        m.name='state_full'
        torch.testing.assert_close(m.step(u,torch.ones_like(a)),torch.ones_like(a))


if __name__=='__main__':
    unittest.main()
