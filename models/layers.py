import torch
import torch.nn as nn

class CustomDropout(nn.Module):
    """Hand-rolled Inverted Dropout implementation."""
    def __init__(self, p: float = 0.5):
        super().__init__()
        if not (0 <= p <= 1):
            raise ValueError("Keep probability must be within [0, 1]")
        self.p = p

    def forward(self, x):
        # Only apply during training and if probability is non-zero
        if not self.training or self.p == 0.0:
            return x
        
        # Generate random values and create binary gate
        noise = torch.rand_like(x)
        binary_mask = (noise > self.p).to(x.dtype)
        
        # Scale remaining values to maintain expected activation magnitude
        return (x * binary_mask) / (1.0 - self.p)

# """Reusable custom layers"""

# import torch
# import torch.nn as nn

# class CustomDropout(nn.Module):
#     """Custom Dropout layer using inverted dropout."""

#     def __init__(self, p: float = 0.5):
#         """
#         Initialize the CustomDropout layer.
#         Args:
#             p: Dropout probability.
#         """
#         super().__init__()
#         if p < 0 or p > 1:
#             raise ValueError("Dropout probability must be between 0 and 1.")
#         self.p = p

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Forward pass for the CustomDropout layer."""
#         if not self.training or self.p == 0.0:
#             return x
        
#         mask = (torch.rand_like(x) > self.p).float()
        
#         return x * mask / (1.0 - self.p)
