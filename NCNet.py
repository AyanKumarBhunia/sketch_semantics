import torch.nn as nn
import torch
# import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


def corr_score(corr_xy):
    corr_B = corr_xy  # .permute(0, 2, 1)
    corr_A = corr_xy.permute(0, 2, 1)

    score_A, _ = corr_A.softmax(dim=-1).max(-1)   # (b, N1, N2).max(2) -> (b, N1, 1)
    score_B, _ = corr_B.softmax(dim=-1).max(-1)   # (b, N2, N1).max(2) -> (b, N2, 1)
    score = (score_A.mean() + score_B.mean()) / 2

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
        mask = max_neg_value(torch.bmm(mask_A, mask_B.transpose(1, 2)))

        if self.normalization:
            correlation_tensor = featureL2Norm(torch.nn.functional.relu(correlation_tensor))

        return correlation_tensor, mask


def maxpool(corr, k_size=4):
    slices = []
    for i in range(k_size):
        for j in range(k_size):
            slices.append(corr[:, 0, i::k_size, j::k_size].unsqueeze(0))
    slices = torch.cat(slices, dim=1)
    corr, max_idx = torch.max(slices, dim=1, keepdim=True)
    max_j = torch.fmod(max_idx, k_size)
    max_i = max_idx.sub(max_j).div(k_size)
    # i,j represent the *relative* coords of the max point in the box of size k_size*k_size
    return (corr, max_i, max_j)


def MutualMatching(corr2d):  #b, N1, N2 is input ..
    # mutual matching
    # batch_size, ch, fs1, fs2, fs3, fs4 = corr4d.size()   # OLD 4D
    # b, N1, N2 =  corr2d.size() in this case

    # corr4d_B = corr4d.view(batch_size, fs1 * fs2, fs3, fs4)  # [batch_idx,k_A,i_B,j_B]   # OLD 4D
    # corr4d_A = corr4d.view(batch_size, fs1, fs2, fs3 * fs4)

    # nothing to modify. Just get the max in each axis.

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

        num_layers = len(kernel_sizes)
        nn_modules = list()
        self.corr.eval()
        for i in range(num_layers):
            if i == 0:
                ch_in = 1
            else:
                ch_in = channels[i-1]
            ch_out = channels[i]
            k_size = kernel_sizes[i]
            nn_modules.append(nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=k_size,
                                        bias=True, padding=1))
            nn_modules.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*nn_modules)
        if use_cuda:
            self.conv.cuda()

    def forward(self, feature_A, feature_B, mask_A, mask_B):
        # feature_A -> b, N1, d
        # feature_B -> b, N2, d

        corr_tensor, mask = self.corr(feature_A, feature_B, mask_A, mask_B)      # equation 1  -- done
        # corr_tensor -> b, N1, N2
        # mask basically takes care of the negative infinity part.
        assert (torch.isfinite(corr_tensor).all())
        # pdb.set_trace()
        corr_tensor = MutualMatching(corr_tensor)                               # equation 4+5

        corr_tensor = corr_tensor.unsqueeze(1)
        if self.symmetric_mode:                                                 # equation 2
            # apply network on the input and its "transpose" (swapping A-B to B-A ordering of the correlation tensor),
            # this second result is "transposed back" to the A-B ordering to match the first result and be able to add together
            corr_tensor = self.conv(corr_tensor) + self.conv(corr_tensor.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
            # because of the ReLU layers in between linear layers,
            # this operation is different than convolution of a single time with the filters+filters^T
            # and therefore it makes sense to do this.
        else:
            corr_tensor = self.conv(corr_tensor)
        if self.pool:
            corr_tensor = maxpool(corr_tensor, k_size=self.k_size)

        corr_tensor = corr_tensor.squeeze(1)

        corr_tensor = MutualMatching(corr_tensor)
        
        # problem lies here

        # assert(torch.isfinite(x).all())
        # r_A = x / x.max(-1, keepdim=True)[0]
        # assert(torch.isfinite(r_A).all()) # This assert fails, commented since Loss becomes NaN
        # r_B = x / x.max(1, keepdim=True)[0]
        # assert(torch.isfinite(r_B).all())
        # x = x * r_A * r_B

        # corr4d = MutualMatching(corr4d)
        # corr4d = self.NeighConsensus(corr4d)
        # corr4d = MutualMatching(corr4d)

        corr_tensor = corr_tensor * mask
        return corr_tensor


