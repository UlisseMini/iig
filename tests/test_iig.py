import torch as t
import iig


class ToyModel(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = t.nn.Parameter(t.tensor([1.0, 2.0]))
        self.w2 = t.nn.Parameter(t.tensor([3.0, 4.0]))

    def forward(self, x):
        x = x * self.w1
        x = x * self.w2
        return x


def test_iig_toy():
    input = t.tensor([[1.0, 2.0]])
    baseline = t.tensor([[0.0, 0.0]])

    model = ToyModel()
    attr, _delta = iig.trace(model, input, baseline, target=0)
    attr = t.tensor(attr[0].detach(), dtype=t.float32)

    assert t.allclose(attr, t.tensor([[3.0, 0.0]])), f"attr={attr} != [[3.0, 0.0]]"
