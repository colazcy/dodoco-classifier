import math
import torch
import torchlibrosa as tl
import torch.nn as nn
import torch.nn.functional as F


class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(bottleneck),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.se(input)
        return input * x


class Bottle2neck(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale=8):
        super(Bottle2neck, self).__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1
        convs = []
        bns = []
        num_pad = math.floor(kernel_size / 2) * dilation
        for i in range(self.nums):
            convs.append(
                nn.Conv1d(
                    width,
                    width,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=num_pad,
                )
            )
            bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.width = width
        self.se = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)

        out = self.se(out)
        out += residual
        return out


class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            "flipped_filter",
            torch.FloatTensor([-self.coef, 1.0]).unsqueeze(0).unsqueeze(0),
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), "reflect")
        return F.conv1d(input, self.flipped_filter).squeeze(1)


class FbankAug(nn.Module):

    def __init__(self, freq_mask_width=(0, 8), time_mask_width=(0, 10)):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        super().__init__()

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        else:
            D = time
            width_range = self.time_mask_width

        mask_len = torch.randint(
            width_range[0], width_range[1], (batch, 1), device=x.device
        ).unsqueeze(2)
        mask_pos = torch.randint(
            0, max(1, D - mask_len.max()), (batch, 1), device=x.device
        ).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)

        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)

        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x):
        x = self.mask_along_axis(x, dim=2)
        x = self.mask_along_axis(x, dim=1)
        return x


# torchlibrosa's MelSpectrogram is different from torchaudio's
class Reshape(nn.Module):
    def __init__(self):
        super(Reshape, self).__init__()

    def forward(self, x):
        x = x.squeeze(1)
        x = x.permute(0, 2, 1)
        return x


class ECAPA_TDNN(nn.Module):

    def __init__(self, sample_rate, C=1024):
        super(ECAPA_TDNN, self).__init__()

        n_fft = 512
        win_length = 400
        hop_length = 160
        n_mels = 80
        f_min = 20
        f_max = 7600
        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),
            tl.Spectrogram(
                center=False,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
            ),
            tl.LogmelFilterBank(
                n_fft=n_fft,
                sr=sample_rate,
                n_mels=n_mels,
                fmin=f_min,
                fmax=f_max,
                is_log=False,
            ),
            Reshape(),
        )

        self.specaug = FbankAug()

        self.conv1 = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        self.layer4 = nn.Conv1d(3 * C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, out_features=192)
        self.bn6 = nn.BatchNorm1d(192)

    def forward(self, x):
        with torch.no_grad():
            x = self.torchfbank(x) + 1e-6
            x = x.log()
            x = x - torch.mean(x, dim=-1, keepdim=True)

            if self.training:
                x = self.specaug(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)

        x = self.layer4(torch.cat((x1, x2, x3), dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        global_x = torch.cat(
            (
                x,
                torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
                torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(
                    1, 1, t
                ),
            ),
            dim=1,
        )

        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-4))

        x = torch.cat((mu, sg), 1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)

        return x
