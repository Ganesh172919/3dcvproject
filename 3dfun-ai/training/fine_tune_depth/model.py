"""Compact RGB-to-depth network for the synthetic 3DFUN AI training experiment."""
import torch
from torch import nn
from torch.nn import functional as F


class TinyDepth(nn.Module):
    """Full-resolution skip connection plus a low-resolution context branch."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 5, padding=2), nn.ReLU())
        self.context = nn.Sequential(
            nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(24, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 24, 3, padding=2, dilation=2), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Conv2d(40, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 1, 1)
        )
        nn.init.constant_(self.head[-1].bias, 1.5)

    def forward(self, rgb):
        detail = self.stem(rgb)
        context = F.interpolate(self.context(detail), size=rgb.shape[-2:],
                                mode="bilinear", align_corners=False)
        return F.softplus(self.head(torch.cat([detail, context], dim=1))) + 1e-3


def load_model(checkpoint, device="cpu"):
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    if state.get("architecture") != "tinydepth-v1":
        raise ValueError("Expected a tinydepth-v1 training checkpoint.")
    model = TinyDepth().to(device)
    model.load_state_dict(state["weights"])
    return model.eval(), state
