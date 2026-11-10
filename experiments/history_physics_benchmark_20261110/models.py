"""Matched-size sequence models; neither receives current ground-truth state."""
import torch
from torch import nn
from torch.nn import functional as F


class PlainGRU(nn.Module):
    def __init__(self, task):
        super().__init__()
        self.task = task
        self.gru = nn.GRU(5, 64, batch_first=True)
        self.head = nn.Linear(64, 1)

    def forward(self, x, initial):
        # The initial state is already encoded identically in every input token.
        return self.head(self.gru(x)[0])


class ConstrainedState(nn.Module):
    def __init__(self, task):
        super().__init__()
        self.task = task
        self.net = nn.Sequential(nn.Linear(6, 112), nn.SiLU(),
                                 nn.Linear(112, 112), nn.SiLU(),
                                 nn.Linear(112, 2 if task == "thermal" else 1))

    def forward(self, x, initial):
        state = initial
        output = []
        for inputs in x.unbind(1):
            rates = self.net(torch.cat((inputs, state), dim=-1))
            if self.task == "thermal":
                # Learned positive loss/source rates. No H/C or exact rate is
                # supplied internally: the network must learn from common inputs.
                k = F.softplus(rates[:, :1]) * 0.03 + 1e-6
                source = F.softplus(rates[:, 1:]) * 0.03 * inputs[:, :1]
                dt = inputs[:, 1:2] * 5.0
                one_minus_decay = -torch.expm1(-k * dt)
                state = state * (1 - one_minus_decay) + source / k * one_minus_decay
            else:
                # Only positivity, zero change without cyclic load, and absorbing
                # failure are encoded. Neither Paris exponent nor crack-size
                # power law is hard-coded into this learned finite-step increment.
                rate = F.softplus(rates) * 0.02
                state = torch.clamp(state + inputs[:, :1] * inputs[:, 1:2] * rate, max=1.0)
            output.append(state)
        return torch.stack(output, dim=1)


def make_model(name, task):
    return {"gru": PlainGRU, "constrained": ConstrainedState}[name](task)
