"""State-feedback factorial and nested constrained-update ablations."""
import torch
from torch import nn
from torch.nn import functional as F

GRU_NAMES = ('gru_plain', 'gru_state', 'gru_projected', 'gru_state_projected')
TASK_MODELS = {
    'thermal': (*GRU_NAMES, 'state_free', 'state_full'),
    'fatigue': (*GRU_NAMES, 'state_free', 'state_gated', 'state_positive', 'state_full', 'state_cubic'),
}


class FactorialGRU(nn.Module):
    """Identical parameters/init; only state input gate and output projection vary.

    A previous output is bookkeeping in every variant. Only the state-feedback
    factor exposes it to the recurrent network as the sixth input feature.
    """
    def __init__(self, name, task):
        super().__init__()
        self.task, self.name = task, name
        self.feedback = name in ('gru_state', 'gru_state_projected')
        self.projected = name in ('gru_projected', 'gru_state_projected')
        self.gru = nn.GRUCell(6, 64)
        self.head = nn.Linear(64, 1)
        # Common initialization, chosen before fitting, avoids all outputs
        # initially landing on a projection boundary with zero gradient.
        nn.init.constant_(self.head.bias, .1 if task == 'thermal' else .2)

    def forward(self, x, initial, *, return_candidates=False):
        state, hidden, out, candidates = initial, None, [], []
        for u in x.unbind(1):
            feature = state if self.feedback else torch.zeros_like(state)
            hidden = self.gru(torch.cat((u, feature), -1), hidden)
            candidate = self.head(hidden)
            if return_candidates:
                candidates.append(candidate)
            if self.projected:
                if self.task == 'thermal':
                    candidate = candidate.clamp_min(0)
                    candidate = torch.where(u[:, :1] == 0, torch.minimum(candidate, state), candidate)
                else:
                    candidate = torch.minimum(torch.maximum(candidate, state), torch.ones_like(state))
                    candidate = torch.where(u[:, :1] == 0, state, candidate)
            state = candidate
            out.append(state)
        result = torch.stack(out, 1)
        return (result, torch.stack(candidates, 1)) if return_candidates else result


class ExplicitState(nn.Module):
    def __init__(self, name, task):
        super().__init__()
        self.name, self.task = name, task
        self.net = nn.Sequential(nn.Linear(6, 112), nn.SiLU(), nn.Linear(112, 112),
                                 nn.SiLU(), nn.Linear(112, 2 if task == 'thermal' else 1))

    def step(self, inputs, state, *, disable_cap=False, rate_scale=1.):
        raw = self.net(torch.cat((inputs, state), -1))
        if self.task == 'thermal':
            dt = inputs[:, 1:2] * 5
            if self.name == 'state_free':
                return state + dt * .03 * (raw[:, :1] + raw[:, 1:] * state)
            k = F.softplus(raw[:, :1]) * .03 + 1e-6
            source = F.softplus(raw[:, 1:]) * .03 * inputs[:, :1]
            decay = -torch.expm1(-k * dt)
            return state * (1 - decay) + source / k * decay
        if self.name in ('state_free', 'state_gated'):
            rate = raw * .02
        else:
            rate = F.softplus(raw) * .02
        gate = 1. if self.name == 'state_free' else inputs[:, :1]
        if self.name == 'state_cubic':
            gate = gate ** 3
        next_state = state + gate * inputs[:, 1:2] * rate * rate_scale
        if self.name in ('state_full', 'state_cubic') and not disable_cap:
            next_state = next_state.clamp_max(1.)
        return next_state

    def forward(self, x, initial, *, disable_cap=False, rate_scale=1.):
        state, out = initial, []
        for u in x.unbind(1):
            state = self.step(u, state, disable_cap=disable_cap, rate_scale=rate_scale)
            out.append(state)
        return torch.stack(out, 1)


def make_model(name, task):
    if name not in TASK_MODELS[task]:
        raise ValueError((name, task))
    return FactorialGRU(name, task) if name in GRU_NAMES else ExplicitState(name, task)
