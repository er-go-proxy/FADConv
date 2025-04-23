import torch
import torch.nn as nn
import torch.nn.functional as F
#---------------------FAT and 2D DCT---------------------
class HierarchicalDCT(nn.Module):
    def __init__(self, low_freq=4, high_freq=0,pool_size=16):
        super().__init__()
        self.pool_size = pool_size
        self.low_freq = low_freq
        self.high_freq = high_freq

        self.register_buffer('dct_matrix', self._init_dct_matrix(pool_size))

        self.pool = nn.AdaptiveAvgPool2d(pool_size)

    def _init_dct_matrix(self, size):
        n = torch.arange(size, dtype=torch.float32)
        k = n.view(-1, 1)
        dct_mat = torch.cos((2 * k + 1) * n * torch.pi / (2 * size))
        scale = 1.0 / size
        dct_mat *= scale
        return dct_mat

    def _batch_dct(self, x):
        # x: [B, C, H, W] (H=W=pool_size)
        return torch.einsum('bchw,hi,wj->bcij', x, self.dct_matrix, self.dct_matrix)

    def forward(self, x):
        x_pooled = self.pool(x)
        dct_low = self._batch_dct(x_pooled)[..., :self.low_freq, :self.low_freq]
        if self.high_freq > 0:
            dct_high = self._batch_dct(x_pooled)[..., -self.high_freq:, -self.high_freq:]
            x_cat = torch.cat([dct_low.flatten(2), dct_high.flatten(2)], dim=2)  # [B,C,(l*l+h*h)]
        else:
            x_cat = dct_low.flatten(2)

        return x_cat

class FADConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_experts,
                 stride=1, padding=0, dilation=1, groups=1, low_freq=4, high_freq=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_experts = num_experts
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.low_freq=low_freq

        self.dct = HierarchicalDCT(low_freq,high_freq)
        self.gap = nn.AdaptiveAvgPool2d(1)
        # expert kernels
        self.experts = nn.Parameter(
            torch.randn(num_experts, out_channels, in_channels // groups, kernel_size, kernel_size)
        )
        nn.init.kaiming_normal_(self.experts, mode='fan_out', nonlinearity='relu')

        self.routing = nn.Sequential(
            nn.Linear(self.in_channels, max(32, in_channels // 16)),
            nn.ReLU(inplace=True),
            nn.Linear(max(32, in_channels // 16), num_experts),
            nn.Softmax(dim=-1)
        )
        self.fusion = nn.Sequential(
            nn.Linear(low_freq**2+high_freq**2, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        freq_features = self.dct(x)  # [B, C, f]
        freq_features =self.fusion(freq_features).squeeze(dim=2)#[B, C]—————replace GAP
        route_weights = self.routing(freq_features)  # [B, num_experts]

        dynamic_kernel = torch.einsum('be,eoihw->boihw', route_weights, self.experts)

        x = x.view(1, -1, H, W)
        dynamic_kernel = dynamic_kernel.view(-1, self.in_channels // self.groups,
                                             self.kernel_size, self.kernel_size)  # [B*out, C, K, K]

        output = F.conv2d(
            x, dynamic_kernel,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=B * self.groups
        )
        x = output.view(B, self.out_channels, output.size(2), output.size(3))

        return x


# Replace the static convolution kernel with FADConv.
# For example：
# FADConv(in_channels, out_channels, 3,stride=stride,padding=1,num_experts=4),
# FADConv(in_channels, out_channels, kernel_size=3,padding=18, dilation=18,num_experts=num_experts)
