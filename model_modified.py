from functools import reduce
from typing import List
import torch
from torch import nn
from torch.nn import functional as fn

class ConvBN(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            dilation: int = 1,
    ) -> None:
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, 1),
            padding='same',
            dilation=dilation,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fn.relu(self.bn(self.conv(x)))

    def reset_parameters(self):
        for _, child in self.named_children():
            if 'reset_parameters' in dir(child):
                child.reset_parameters()


class InnerUUnit(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            middle_channels: int,
            kernel_size: int,
            pooling_size: int,
            sq_len: int,
            depth: int = 4,
    ) -> None:
        super(InnerUUnit, self).__init__()
        self.enc0 = ConvBN(in_channels, middle_channels, kernel_size)
        self.encs = nn.ModuleList()
        self.decs = nn.ModuleList()
        self.sq_len = sq_len
        self.depth = depth

        for i in range(depth):
            if i == 0:
                self.encs.append(
                    ConvBN(middle_channels, middle_channels, kernel_size)
                )
            else:
                self.encs.append(nn.Sequential(
                    nn.MaxPool2d((pooling_size, 1)),
                    ConvBN(middle_channels, middle_channels, kernel_size)
                ))

        for i in range(depth):
            if i == depth-1:
                self.decs.append(
                    ConvBN(middle_channels * 2, out_channels, kernel_size)
                )
            else:
                self.decs.append(nn.Sequential(
                    ConvBN(
                        middle_channels * 2,
                        middle_channels,
                        kernel_size
                    ),
                    nn.Upsample(
                        size=(
                            self.sq_len // (pooling_size ** (depth - i - 2)),
                            1
                        ),
                        mode='bilinear',
                        align_corners=True
                    )
                ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc0(x)
        x0 = x.clone()
        enc_list = []
        for ly in self.encs:
            x = ly(x)
            enc_list.append(x.clone())
        for (xi, ly) in zip(enc_list[::-1], self.decs):
            x = ly(torch.cat((x, xi), dim=1))

        return torch.cat((x, x0), dim=1)

    def reset_parameters(self):
        for _, child in self.named_children():
            if 'reset_parameters' in dir(child):
                child.reset_parameters()


class MSE(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            dilation_rates: List[int]
    ):
        super(MSE, self).__init__()
        self.dilation_list = nn.ModuleList([
            ConvBN(in_channels, out_channels, kernel_size, d)
            for d in dilation_rates
        ])
        self.down1 = ConvBN(out_channels * 4, out_channels * 2, kernel_size)
        self.down2 = ConvBN(out_channels * 2, out_channels, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = reduce(
            lambda a, b: torch.cat((a, b), dim=1),
            [ly(x.clone()) for ly in self.dilation_list]
        )
        x = self.down1(x)
        x = self.down2(x)
        return x

    def reset_parameters(self):
        for _, child in self.named_children():
            if 'reset_parameters' in dir(child):
                child.reset_parameters()


class SingleStream(nn.Module):
    def __init__(
            self,
            sq_len: int
    ) -> None:

        super(SingleStream, self).__init__()
        self.e1 = InnerUUnit(1, 16, 8, 5, 10, sq_len)
        self.dimension1 = ConvBN(16+8, 8, 1)
        self.down1 = nn.MaxPool2d((10, 1))

        self.e2 = InnerUUnit(8, 32, 8, 5, 8, sq_len // 10)
        self.dimension2 = ConvBN(32+8, 16, 1)
        self.down2 = nn.MaxPool2d((8, 1))

        self.e3 = InnerUUnit(16, 64, 8, 5, 6, sq_len // 10 // 8)
        self.dimension3 = ConvBN(64+8, 32, 1)
        self.down3 = nn.MaxPool2d((6, 1))

        self.e4 = InnerUUnit(32, 128, 8, 5, 4, sq_len // 10 // 8 // 6)
        self.dimension4 = ConvBN(128+8, 64, 1)
        self.down4 = nn.MaxPool2d((4, 1))

        self.e5 = InnerUUnit(64, 256, 8, 5, 2, sq_len // 10 // 8 // 6 // 4)
        self.dimension5 = ConvBN(256+8, 128, 1)
        # 10,128,54,1
        self.mse1 = MSE(8, 32, 5, [1, 2, 3, 4])
        self.mse2 = MSE(16, 24, 5, [1, 2, 3, 4])
        self.mse3 = MSE(32, 16, 5, [1, 2, 3, 4])
        self.mse4 = MSE(64, 8, 5, [1, 2, 3, 4])
        self.mse5 = MSE(128, 5, 5, [1, 2, 3, 4])

        self.up4 = nn.Upsample(
            size=(sq_len // 10 // 8 // 6, 1),
            mode='bilinear',
            align_corners=True
        )
        self.d4 = InnerUUnit(128+8, 64, 8, 5, 4, sq_len // 10 // 8 // 6)

        self.up3 = nn.Upsample(
            size=(sq_len // 10 // 8, 1),
            mode='bilinear',
            align_corners=True
        )
        self.d3 = InnerUUnit(64+16+8, 32, 8, 5, 6, sq_len // 10 // 8)

        self.up2 = nn.Upsample(
            size=(sq_len // 10, 1),
            mode='bilinear',
            align_corners=True
        )
        self.d2 = InnerUUnit(32+24+8, 16, 8, 5, 8, sq_len // 10)

        self.up1 = nn.Upsample(
            size=(sq_len, 1),
            mode='bilinear',
            align_corners=True
        )
        self.d1 = InnerUUnit(16+32+8, 16, 8, 5, 10, sq_len)
        self.last_dimension = ConvBN(16+8, 16, 1)
        self.conv1 = ConvBN(128, 16, 1)
        self.u = nn.Upsample((105000, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.dimension1(self.e1(x))
        x = self.down1(x1.clone())

        x2 = self.dimension2(self.e2(x))
        x = self.down2(x2.clone())

        x3 = self.dimension3(self.e3(x))
        x = self.down3(x3.clone())

        x4 = self.dimension4(self.e4(x))
        x = self.down4(x4.clone())
        x = self.dimension5(self.e5(x))
        x = self.u(self.conv1(x))
        return x

    def reset_parameters(self):
        for _, child in self.named_children():
            if 'reset_parameters' in dir(child):
                child.reset_parameters()

"""
a simplified version of SalientSleepNet as the backbone network for pre-training, 
where both the MSE and MMA modules are removed.
"""
class SleepTTA(nn.Module):
    def __init__(
            self,
            window_size: int = 35,
            channel:str = 'eeg'
    ):
        super(SleepTTA, self).__init__()
        self.l = 3000
        # [*, 1, 1, ws * 3000] -> [*, 16, 1, ws * 3000]
        self.eeg_stream = SingleStream(self.l * window_size)
        self.eog_stream = SingleStream(self.l * window_size)

        self.global_padding1 = nn.AvgPool2d((self.l * window_size, 1))
        self.fc1 = nn.Linear(16, 4)
        self.fc2 = nn.Linear(4, 16)

        self.global_padding2 = nn.AvgPool2d((1, self.l))
        self.conv1 = nn.Conv2d(16, 16, (1, 1), padding='same')
        self.conv2 = nn.Conv2d(16, 5, (5, 1), padding='same')
        self.channel = channel

    def forward(self, x: torch.Tensor,feature_extract=False) -> torch.Tensor:
        eeg, eog = x[..., 0].clone(), x[..., 1].clone()
        eeg = eeg.view(
            eeg.shape[0], eeg.shape[1], eeg.shape[2] * eeg.shape[3], 1
        )
        eog = eog.view(
            eog.shape[0], eog.shape[1], eog.shape[2] * eog.shape[3], 1
        )

        eeg, eog = self.eeg_stream(eeg), self.eog_stream(eog)

        if self.channel == 'eeg':
            merge = eeg
        elif self.channel =='eog':
            merge = eog
        merge = merge.view(
            merge.shape[0], merge.shape[1], merge.shape[2] // self.l, self.l
        )
        if feature_extract:
            merge2=merge.transpose(1,2)
            return merge2
        merge = self.conv1(merge)
        merge = self.global_padding2(merge)
        merge = self.conv2(merge)
        return merge.squeeze(dim=-1)

    def reset_parameters(self):
        for _, child in self.named_children():
            if 'reset_parameters' in dir(child):
                child.reset_parameters()



if __name__ == '__main__':
    from torchinfo import summary
    batch_size =10
    seq_len = 35
    model = SleepTTA(seq_len)
    summary(model, input_size=(batch_size, 1, seq_len, 3000, 2))


