import torch
import torch.nn as nn


class FDAF(nn.Module):
    """
    Frequency-Domain Attention Fusion (FDAF) block.
    This is the core adaptive spectral gating logic used in the project.
    """

    def __init__(self, conv_cls, c1, c2, use_gate=True):
        """
        Args:
            conv_cls: project Conv module class (e.g., ultralytics Conv)
            c1: input channels
            c2: output channels
        """
        super().__init__()
        self.conv1 = conv_cls(c1 * 2, c2, 1, 1)
        self.use_gate = use_gate
        # Learnable spectral gate in frequency domain
        self.gate = nn.Conv2d(c1, c1, 1, 1, 0) if use_gate else None

    def forward(self, x):
        # FFT to frequency domain
        f_x = torch.fft.fft2(x, norm="ortho")
        if self.gate is None:
            f_mod = f_x
        else:
            mag = torch.abs(f_x)
            # Adaptive spectral gating (soft mask in [0, 1])
            w_gate = torch.sigmoid(self.gate(mag))
            f_mod = f_x * w_gate

        # Inverse FFT back to spatial domain
        y_spa = torch.fft.ifft2(f_mod, norm="ortho").real

        # Cross-domain fusion + residual
        return self.conv1(torch.cat([x, y_spa], dim=1)) + x
