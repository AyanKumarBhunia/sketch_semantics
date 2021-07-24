import pdb
import torch.nn as nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn.functional as F


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


def corr_score(corr_xy, len_A, len_B):
    corr_A = corr_xy  # .permute(0, 2, 1)
    corr_B = corr_xy.permute(0, 2, 1)

    score_A_mean = 0
    for x, y, z in zip(corr_A, len_A, len_B):
        score_A_ab = x[:y, :z]
        if (torch.isinf(score_A_ab).sum()) !=0:
            pdb.set_trace()
        score_A_mean += F.softmax(score_A_ab, dim=-1).max(-1)[0].mean()

    score_B_mean = 0
    for x, y, z in zip(corr_B, len_B, len_A):
        score_B_ab = x[:y, :z]
        if (torch.isinf(score_B_ab).sum()) != 0:
            pdb.set_trace()
        score_B_mean += F.softmax(score_B_ab, dim=-1).max(-1)[0].mean()

    score = (score_A_mean + score_B_mean) / 2
    return score


def featureL2Norm(feature):
    epsilon = 1e-6
    norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1)  # .expand_as(feature)
    return torch.div(feature, norm)


class FeatureCorrelation(torch.nn.Module):

    def __init__(self, normalization=True):
        super(FeatureCorrelation, self).__init__()
        self.normalization = normalization

    def forward(self, feature_A, feature_B, mask_A, mask_B):
        # perform matrix mult.
        # (b, N1, d) @ (b, d, N2) -> (b, N1, N2)
        feature_B = feature_B.transpose(1, 2)
        correlation_tensor = torch.bmm(feature_A, feature_B)           # equation 1

        if self.normalization:
            correlation_tensor = featureL2Norm(torch.nn.functional.relu(correlation_tensor))

        return correlation_tensor


def MutualMatching(corr2d):  # corr2d = [b, N1, N2]
    # mutual matching
    # enforced reciprocity. given i-th row -- if argmin is j <===> given j-th col -- argmin is i
    # argmin of axis 2 w.rt. axis 1 should be equal to argmin of axis 1 w.r.t axis 2

    corr2d_N1_max, _ = torch.max(corr2d, dim=1, keepdim=True)    # along N1 axis
    corr2d_N2_max, _ = torch.max(corr2d, dim=2, keepdim=True)    # along N2 axis

    eps = 1e-5
    corr2d_N1 = corr2d / (corr2d_N1_max + eps)  # equation 5
    corr2d_N2 = corr2d / (corr2d_N2_max + eps)

    corr2d = corr2d * (corr2d_N1 * corr2d_N2)  # parenthesis are important for symmetric output  # equation 4

    return corr2d

class NC_Conv2D_Masked(nn.Module):
    def __init__(self, ch_hidden=10, k_size=3, ):
        super(NC_Conv2D_Masked, self).__init__()

        self.conv1 = nn.Conv2d(1, out_channels=ch_hidden, kernel_size=k_size, bias=True, padding=1)
        self.conv2 = nn.Conv2d(ch_hidden, out_channels=ch_hidden, kernel_size=k_size, bias=True, padding=1)
        self.conv3 = nn.Conv2d(ch_hidden, out_channels=1, kernel_size=k_size, bias=True, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x, corr_mask_conv):              # corr_mask_conv = batch, 1, N1, N2
        x = self.relu(self.conv1(x)) * corr_mask_conv  # batch, 10, N1, N2
        x = self.relu(self.conv2(x)) * corr_mask_conv
        x = self.relu(self.conv3(x)) * corr_mask_conv

        return x


class NeighbourhoodConsensus2D(nn.Module):
    def __init__(self, use_conv=True, pool=False, k_size=None, use_cuda=True,
                 kernel_sizes=[3, 3, 3], channels=[10, 10, 1], symmetric_mode=True):
        super(NeighbourhoodConsensus2D, self).__init__()
        self.symmetric_mode = symmetric_mode
        self.kernel_sizes = kernel_sizes
        self.channels = channels
        self.pool = pool
        self.k_size = k_size
        self.corr = FeatureCorrelation()
        self.use_conv = use_conv

        self.masked_conv = NC_Conv2D_Masked()
        if use_cuda:
            self.masked_conv.cuda()

    def forward(self, feature_A, feature_B, mask_A, mask_B):  # A-> b, N1, d ; B-> b, N2, d

        corr_tensor = self.corr(feature_A, feature_B, mask_A, mask_B)      # equation 1   [b, N1, N2]
        corr_mask = torch.bmm(mask_A,mask_B.permute(0, 2, 1))
        corr_mask = corr_mask / 512.0                       # only 1s and 0s

        assert (torch.isfinite(corr_tensor).all())
        corr_tensor = MutualMatching(corr_tensor)                               # equation 4+5

        corr_tensor = corr_tensor.unsqueeze(1)
        corr_mask = corr_mask.unsqueeze(1)
        if self.symmetric_mode:
            corr_tensor = self.masked_conv(corr_tensor, corr_mask) + \
                self.masked_conv(corr_tensor.permute(0, 1, 3, 2), corr_mask.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        else:
            corr_tensor = self.masked_conv(corr_tensor)
        corr_tensor = corr_tensor.squeeze(1)
        corr_mask = corr_mask.squeeze(1)

        corr_tensor = MutualMatching(corr_tensor)                               # equation 4+5

        # https://medium.com/analytics-vidhya/masking-in-transformers-self-attention-mechanism-bad3c9ec235c
        # MASKING to deal with softmax -- summation over softmax output = 1.
        corr_mask = torch.where(corr_mask==1.0, torch.tensor(0.0).to(device), torch.tensor(-float('inf')).to(device))
        corr_tensor = corr_tensor + corr_mask

        return corr_tensor


