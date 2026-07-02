import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    """LoRA wrapper around a frozen Linear layer.

    h = base(x) + (x @ A.T @ B.T) * (alpha / r)
    """

    def __init__(self, base_linear, r=8, alpha=16):
        super().__init__()
        self.base = base_linear
        self.base.requires_grad_(False)
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_dim = base_linear.in_features
        out_dim = base_linear.out_features
        self.lora_A = nn.Parameter(torch.randn(r, in_dim) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_dim, r))

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


def inject_lora(model, r=8, alpha=16):
    """Inject LoRA into all CausalAttention layers of GPT."""
    for name, module in model.named_modules():
        if hasattr(module, "W_q") and hasattr(module, "W_k") and \
           hasattr(module, "W_v") and hasattr(module, "W_o"):
            module.W_q = LoRALayer(module.W_q, r, alpha)
            module.W_k = LoRALayer(module.W_k, r, alpha)
            module.W_v = LoRALayer(module.W_v, r, alpha)
            module.W_o = LoRALayer(module.W_o, r, alpha)
    return model


def lora_params_count(model):
    """Count only the LoRA A/B parameters."""
    return sum(p.numel() for name, p in model.named_parameters()
               if "lora_A" in name or "lora_B" in name)


def freeze_all_except_lora(model):
    """Freeze all parameters; only LoRA A/B remain trainable."""
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)
