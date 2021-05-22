import torch.nn as nn
import torchvision.models as backbone_
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loss(corr_xy, corr_xneg):
    # Positive match

    corr_B = corr_xy  # .permute(0, 2, 1)
    corr_A = corr_xy.permute(0, 2, 1)

    # (b, s1, s2).max(2) -> (b, s1, 1)
    score_A, _ = corr_A.softmax(dim=-1).max(-1)
    # (b, s2, s1).max(2) -> (b, s2, 1)
    score_B, _ = corr_B.softmax(dim=-1).max(-1)
    pos_score = (score_A.mean() + score_B.mean()) / 2

    # Negative match
    corr_B = corr_xneg  # .permute(0, 2, 1)
    corr_A = corr_xneg.permute(0, 2, 1)

    score_A, _ = corr_A.softmax(dim=-1).max(-1)
    score_B, _ = corr_B.softmax(dim=-1).max(-1)
    neg_score = (score_A.mean() + score_B.mean()) / 2

    return neg_score - pos_score


def featureL2Norm(feature):
    epsilon = 1e-6
    norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) +
                     epsilon, 0.5).unsqueeze(1)  # .expand_as(feature)
    return torch.div(feature, norm)


class FeatureCorrelation(torch.nn.Module):
    def __init__(self, normalization=True):
        super(FeatureCorrelation, self).__init__()
        self.normalization = normalization

    def forward(self, feature_A, feature_B, mask_A, mask_B):
        assert(len(feature_A.shape) == 3)
        assert(len(feature_B.shape) == 3)

        b, s1, d = feature_A.size()
        b, s2, d = feature_B.size()
        # reshape features for matrix multiplication
        # (b, s2, d) -> (b, d, s2)
        feature_B = (feature_B * mask_B).transpose(1, 2)
        feature_A = feature_A * mask_A
        # perform matrix mult.

        # (b, s1, d) @ (b, d, s2) -> (b, s1, s2)
        correlation_tensor = torch.bmm(feature_A, feature_B)

        if self.normalization:
            correlation_tensor = featureL2Norm(F.relu(correlation_tensor))

        return correlation_tensor


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


class NeighbourhoodConsensus2D(nn.Module):
    def __init__(self, use_conv=True, pool=False, k_size=None, use_cuda=True, kernel_sizes=[3, 3, 3], channels=[10, 10, 1], symmetric_mode=True):
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
        for i in range(num_layers):
            if i == 0:
                ch_in = 1
            else:
                ch_in = channels[i-1]
            ch_out = channels[i]
            k_size = kernel_sizes[i]
            nn_modules.append(nn.Conv2d(
                in_channels=ch_in, out_channels=ch_out, kernel_size=k_size, bias=True, padding=1))
            nn_modules.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*nn_modules)
        if use_cuda:
            self.conv.cuda()

    def forward(self, feature_A, feature_B, mask_A, mask_B):
        # feature_A -> b, s1, d
        # feature_B -> b, s2, d

        x = self.corr(feature_A, feature_B, mask_A, mask_B)
        # x -> b, s1, s2

        if False:
            x = x.unsqueeze(1)
            if self.symmetric_mode:
                # apply network on the input and its "transpose" (swapping A-B to B-A ordering of the correlation tensor),
                # this second result is "transposed back" to the A-B ordering to match the first result and be able to add together
                x = self.conv(x) + self.conv(x.permute(0, 1, 3, 2)
                                             ).permute(0, 1, 3, 2)
                # because of the ReLU layers in between linear layers,
                # this operation is different than convolving a single time with the filters+filters^T
                # and therefore it makes sense to do this.
            else:
                x = self.conv(x)
            if self.pool:
                x = maxpool(x, k_size=self.k_size)
            return x.squeeze(1)
        # else:
        #     x = x + x.t()  # Modify Eq. 2 to work with (b, s1, s2) & (b, s1, s2)

        # print(x[0, 2, 3], x[0, 3, 2])
        # print(x[0, 1, 3], x[0, 3, 1])
        # print(x[0, 0, 3], x[0, 3, 0])

        r_A = x / x.max(-1, keepdim=True)[0]
        r_B = x / x.max(1, keepdim=True)[0]
        x = x * r_A * r_B
        return x


class Stroke_Embedding_Network(nn.Module):
    def __init__(self, hp):
        super(Stroke_Embedding_Network, self).__init__()

        if hp.data_encoding_type == '3point':
            inp_dim = 3
        elif hp.data_encoding_type == '5point':
            inp_dim = 5
        else:
            raise ValueError('invalid option. Select either 3point/5point')

        self.LSTM_stroke = nn.LSTM(inp_dim, hp.hidden_size,
                                   num_layers=hp.stroke_LSTM_num_layers,
                                   dropout=hp.dropout_stroke,
                                   batch_first=True, bidirectional=True)

        self.embedding_1 = nn.Linear(
            hp.hidden_size*2*hp.stroke_LSTM_num_layers, hp.hidden_size)

        self.LSTM_global = nn.LSTM(hp.hidden_size, hp.hidden_size, num_layers=hp.stroke_LSTM_num_layers,
                                   dropout=hp.dropout_stroke,
                                   batch_first=True, bidirectional=True)

        self.embedding_2 = nn.Linear(2*hp.hidden_size, hp.hidden_size)
        self.layernorm = nn.LayerNorm(hp.hidden_size)

    def forward(self, batch, type='anchor'):

        if type == 'anchor':
            # batch['stroke_wise_split'][:,:,:2] /= 800
            x = pack_padded_sequence(batch['stroke_wise_split_anchor'].to(device),
                                     batch['every_stroke_len_anchor'],
                                     batch_first=True, enforce_sorted=False)
            _, (x_stroke, _) = self.LSTM_stroke(x.float())
            x_stroke = x_stroke.permute(1, 0, 2).reshape(x_stroke.shape[1], -1)
            x_stroke = self.embedding_1(x_stroke)

            x_sketch = x_stroke.split(batch['num_stroke_per_anchor'])
            x_sketch_h = x_sketch
            x_sketch = pad_sequence(x_sketch, batch_first=True)

            x_sketch = pack_padded_sequence(x_sketch, torch.tensor(batch['num_stroke_per_anchor']),
                                            batch_first=True, enforce_sorted=False)
            _, (x_sketch_hidden, _) = self.LSTM_global(x_sketch.float())
            x_sketch_hidden = x_sketch_hidden.permute(
                1, 0, 2).reshape(x_sketch_hidden.shape[1], -1)
            x_sketch_hidden = self.embedding_2(x_sketch_hidden)

            out = []
            for x, y in zip(x_sketch_h, x_sketch_hidden):
                out.append(self.layernorm(x + y))
            out, num_stroke_list = pad_sequence(
                out, batch_first=True), batch['num_stroke_per_anchor']

        elif type == 'positive':

            x = pack_padded_sequence(batch['stroke_wise_split_positive'].to(device),
                                     batch['every_stroke_len_positive'],
                                     batch_first=True, enforce_sorted=False)
            _, (x_stroke, _) = self.LSTM_stroke(x.float())
            x_stroke = x_stroke.permute(1, 0, 2).reshape(x_stroke.shape[1], -1)
            x_stroke = self.embedding_1(x_stroke)

            x_sketch = x_stroke.split(batch['num_stroke_per_positive'])
            x_sketch_h = x_sketch
            x_sketch = pad_sequence(x_sketch, batch_first=True)

            x_sketch = pack_padded_sequence(x_sketch, torch.tensor(batch['num_stroke_per_positive']),
                                            batch_first=True, enforce_sorted=False)
            _, (x_sketch_hidden, _) = self.LSTM_global(x_sketch.float())
            x_sketch_hidden = x_sketch_hidden.permute(
                1, 0, 2).reshape(x_sketch_hidden.shape[1], -1)
            x_sketch_hidden = self.embedding_2(x_sketch_hidden)

            out = []
            for x, y in zip(x_sketch_h, x_sketch_hidden):
                out.append(self.layernorm(x + y))
            out, num_stroke_list = pad_sequence(
                out, batch_first=True), batch['num_stroke_per_positive']

        elif type == 'negative':

            x = pack_padded_sequence(batch['stroke_wise_split_negative'].to(device),
                                     batch['every_stroke_len_negative'],
                                     batch_first=True, enforce_sorted=False)
            _, (x_stroke, _) = self.LSTM_stroke(x.float())
            x_stroke = x_stroke.permute(1, 0, 2).reshape(x_stroke.shape[1], -1)
            x_stroke = self.embedding_1(x_stroke)

            x_sketch = x_stroke.split(batch['num_stroke_per_negative'])
            x_sketch_h = x_sketch
            x_sketch = pad_sequence(x_sketch, batch_first=True)

            x_sketch = pack_padded_sequence(x_sketch, torch.tensor(batch['num_stroke_per_negative']),
                                            batch_first=True, enforce_sorted=False)
            _, (x_sketch_hidden, _) = self.LSTM_global(x_sketch.float())
            x_sketch_hidden = x_sketch_hidden.permute(
                1, 0, 2).reshape(x_sketch_hidden.shape[1], -1)
            x_sketch_hidden = self.embedding_2(x_sketch_hidden)

            out = []
            for x, y in zip(x_sketch_h, x_sketch_hidden):
                out.append(self.layernorm(x + y))
            out, num_stroke_list = pad_sequence(
                out, batch_first=True), batch['num_stroke_per_negative']

        return out, num_stroke_list


class VGG_Network(nn.Module):
    def __init__(self, hp):
        super(VGG_Network, self).__init__()
        backbone = backbone_.vgg16(pretrained=True)  # vgg16, vgg19_bn

        self.features = backbone.features
        self.avgpool = backbone.avgpool
        backbone.classifier._modules['6'] = nn.Linear(4096, 250)
        self.classifier = backbone.classifier

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Resnet_Network(nn.Module):
    def __init__(self, hp):
        super(Resnet_Network, self).__init__()
        # resnet50, resnet18, resnet34
        backbone = backbone_.resnet50(pretrained=True)

        self.features = nn.Sequential()
        for name, module in backbone.named_children():
            if name not in ['avgpool', 'fc']:
                self.features.add_module(name, module)

        self.pool_method = nn.AdaptiveMaxPool2d(1)  # as default

        if hp.dataset_name == 'TUBerlin':
            num_class = 250
        else:
            num_class = 125

        self.classifier = nn.Linear(2048, num_class)

    def forward(self, input, bb_box=None):
        x = self.features(input)
        x = self.pool_method(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
