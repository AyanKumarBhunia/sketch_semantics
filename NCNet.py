import torch.nn as nn
import torch
# import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn.functional as F

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

def corr_score_us(corr_xy, len_A, len_B):
    corr_B = corr_xy  # .permute(0, 2, 1)
    corr_A = corr_xy.permute(0, 2, 1)

    # score_A, _ = corr_A.softmax(dim=-1).max(-1)   # (b, N1, N2).max(2) -> (b, N1, 1)
    # score_B, _ = corr_B.softmax(dim=-1).max(-1)   # (b, N2, N1).max(2) -> (b, N2, 1)

    score_A_mean = 0
    for x, y, z in zip(corr_B, len_A, len_B):
        score_A_ab = x[:y, :z]
        print(torch.isinf(score_A_ab).sum())
        score_A_mean += F.softmax(score_A_ab, dim=-1).max(-1)[0].mean()

    score_B_mean = 0
    for x, y, z in zip(corr_A, len_B, len_A):
        score_B_ab = x[:y, :z]
        print(torch.isinf(score_B_ab).sum())
        score_B_mean += F.softmax(score_B_ab, dim=-1).max(-1)[0].mean()

    score = (score_A_mean + score_B_mean) / 2
    print(score)

    # corr_B: batch, N1_max, N2_max
    return score


def corr_score(corr_xy):
    corr_B = corr_xy  # .permute(0, 2, 1)
    corr_A = corr_xy.permute(0, 2, 1)

    corr_A.softmax(dim=-1)

    score_A, _ = corr_A.softmax(dim=-1).max(-1)   # (b, N1, N2).max(2) -> (b, N1, 1)
    score_B, _ = corr_B.softmax(dim=-1).max(-1)   # (b, N2, N1).max(2) -> (b, N2, 1)


    print(torch.nansum(score_A), torch.nansum(score_B))
    # score_A_mean = torch.nansum(score_A)/ (1 - torch.isnan(score_A).float()).sum()
    # score_B_mean = torch.nansum(score_B) / (1 - torch.isnan(score_B).float()).sum()


    score_A_mean = (torch.where(torch.isnan(score_A), torch.tensor(0.0).to(device), score_A)).sum() / (1 - torch.isnan(score_A).float()).sum()
    score_B_mean = (torch.where(torch.isnan(score_B), torch.tensor(0.0).to(device), score_B)).sum() / (1 - torch.isnan(score_B).float()).sum()

    # https://discuss.pytorch.org/t/torch-nansum-yields-nan-gradients-unexpectedly/117115/3

    print(score_A_mean, score_B_mean)
    score = (score_A_mean + score_B_mean) / 2

    print(score)
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
        assert(len(feature_A.shape) == 3)
        assert(len(feature_B.shape) == 3)

        # taking 3D version of NCNets
        b, N1, d = feature_A.size()
        b, N2, d = feature_B.size()
        # reshape features for matrix multiplication
        # (b, N2, d) -> (b, d, N2)
        feature_B = feature_B.transpose(1, 2)
        # feature_A = feature_A

        # perform matrix mult.
        # (b, N1, d) @ (b, d, N2) -> (b, N1, N2)
        correlation_tensor = torch.bmm(feature_A, feature_B)           # equation 1
        # mask = max_neg_value(torch.bmm(mask_A, mask_B.transpose(1, 2)))

        if self.normalization:
            correlation_tensor = featureL2Norm(torch.nn.functional.relu(correlation_tensor))

        return correlation_tensor


def MutualMatching(corr2d):  #b, N1, N2 is input ..
    # mutual matching
    # enforced reciprocity. given i-th row -- if argmin is j <===> given j-th col -- argmin is i
    # argmin of axis 2 w.rt. axis 1 should be equal to argmin of axis 1 w.r.t axis 2



    # get max
    corr2d_N1_max, _ = torch.max(corr2d, dim=1, keepdim=True)    # along N1 axis
    corr2d_N2_max, _ = torch.max(corr2d, dim=2, keepdim=True)    # along N2 axis

    eps = 1e-5
    corr2d_N1 = corr2d / (corr2d_N1_max + eps)  # equation 5
    corr2d_N2 = corr2d / (corr2d_N2_max + eps)

    # corr4d_B = corr4d_B.view(batch_size, 1, fs1, fs2, fs3, fs4)  # doesn't require realignment
    # corr4d_A = corr4d_A.view(batch_size, 1, fs1, fs2, fs3, fs4)  # OLD 4D

    corr2d = corr2d * (corr2d_N1 * corr2d_N2)  # parenthesis are important for symmetric output  # equation 4

    return corr2d

class NC_Conv2D_Masked(nn.Module):
    def __init__(self, ch_hidden=10, k_size=3):
        super(NC_Conv2D_Masked, self).__init__()

        self.conv1 = nn.Conv2d(1, out_channels=ch_hidden, kernel_size=k_size,
                                bias=True, padding=1)
        self.conv2 = nn.Conv2d(ch_hidden, out_channels=ch_hidden, kernel_size=k_size,
                                bias=True, padding=1)
        self.conv3 = nn.Conv2d(ch_hidden, out_channels=1, kernel_size=k_size,
                                bias=True, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        mask_cal = torch.ones_like(x)
        mask_cal[x == 0] = 0. # batch, 1, N1, N2

        x = self.relu(self.conv1(x)) * mask_cal # batch, 10, N1, N2
        x = self.relu(self.conv2(x)) * mask_cal
        x = self.relu(self.conv3(x)) * mask_cal

        return x


class NeighbourhoodConsensus2D(nn.Module):
    def __init__(self, use_conv=True, pool=False, k_size=None, use_cuda=True,
                 kernel_sizes=[3, 3, 3], channels=[10, 10, 1], symmetric_mode=False):
        super(NeighbourhoodConsensus2D, self).__init__()
        self.symmetric_mode = symmetric_mode
        self.kernel_sizes = kernel_sizes
        self.channels = channels
        self.pool = pool
        self.k_size = k_size
        self.corr = FeatureCorrelation()
        self.use_conv = use_conv

        # self.masked_conv = NC_Conv2D_Masked()
        # if use_cuda:
        #     self.masked_conv.cuda()

    def forward(self, feature_A, feature_B, mask_A, mask_B):
        # feature_A -> b, N1, d
        # feature_B -> b, N2, d

        corr_tensor = self.corr(feature_A, feature_B, mask_A, mask_B)      # equation 1  -- done
        # corr_tensor -> b, N1, N2
        # mask basically takes care of the negative infinity part.
        assert (torch.isfinite(corr_tensor).all())
        # pdb.set_trace()
        corr_tensor = MutualMatching(corr_tensor)                               # equation 4+5

        corr_tensor = corr_tensor.unsqueeze(1)

        # if self.symmetric_mode:                                                 # equation 2
        #     # apply network on the input and its "transpose" (swapping A-B to B-A ordering of the correlation tensor),
        #     # this second result is "transposed back" to the A-B ordering to match the first result and be able to add together
        #     corr_tensor = self.masked_conv(corr_tensor) + self.masked_conv(corr_tensor.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        #     # because of the ReLU layers in between linear layers,
        #     # this operation is different than convolution of a single time with the filters+filters^T
        #     # and therefore it makes sense to do this.
        # else:
        #     corr_tensor = self.masked_conv(corr_tensor)


        corr_tensor = corr_tensor.squeeze(1)

        corr_tensor = MutualMatching(corr_tensor)

        # https://medium.com/analytics-vidhya/masking-in-transformers-self-attention-mechanism-bad3c9ec235c
        # MASKING to deal with softmax -- summation over softmax output = 1.

        mask = torch.zeros_like(corr_tensor)
        mask[corr_tensor == 0] = -float('inf')
        corr_tensor_masked = corr_tensor + mask

        # mask_index = [torch.where(m==0)[0] for m in mask]

        #  masking strategy 1: -float('inf') followed by softmax
        #  masking strategy 1: float('nan') followed by torch.nansum()

        return corr_tensor_masked


