import torch.nn as nn
import torchvision.models as backbone_
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

        self.embedding_1 = nn.Linear(hp.hidden_size*2*hp.stroke_LSTM_num_layers, hp.hidden_size)

        self.LSTM_global = nn.LSTM(hp.hidden_size, hp.hidden_size, num_layers=hp.stroke_LSTM_num_layers,
                dropout=hp.dropout_stroke,
                batch_first = True, bidirectional=True)

        self.embedding_2 = nn.Linear(2*hp.hidden_size, hp.hidden_size)
        self.layernorm = nn.LayerNorm(hp.hidden_size)

    def forward(self, batch, type = 'anchor'):

        if type == 'anchor':
            # batch['stroke_wise_split'][:,:,:2] /= 800
            x = pack_padded_sequence(batch['stroke_wise_split_anchor'].to(device),
                    batch['every_stroke_len_anchor'],
                    batch_first=True, enforce_sorted=False)
            _, (x_stroke, _) = self.LSTM_stroke(x.float())
            x_stroke = x_stroke.permute(1,0,2).reshape(x_stroke.shape[1], -1)
            x_stroke = self.embedding_1(x_stroke)

            x_sketch = x_stroke.split(batch['num_stroke_per_anchor'])
            x_sketch_h = x_sketch
            x_sketch = pad_sequence(x_sketch, batch_first=True)

            x_sketch = pack_padded_sequence(x_sketch, torch.tensor(batch['num_stroke_per_anchor']),
                                            batch_first=True, enforce_sorted=False)
            _, (x_sketch_hidden, _) = self.LSTM_global(x_sketch.float())
            x_sketch_hidden = x_sketch_hidden.permute(1, 0, 2).reshape(x_sketch_hidden.shape[1], -1)
            x_sketch_hidden = self.embedding_2(x_sketch_hidden)

            out = []
            for x, y in zip(x_sketch_h, x_sketch_hidden):
                out.append(self.layernorm(x + y))
            out, num_stroke_list = pad_sequence(out, batch_first=True), batch['num_stroke_per_anchor']

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
            x_sketch_hidden = x_sketch_hidden.permute(1, 0, 2).reshape(x_sketch_hidden.shape[1], -1)
            x_sketch_hidden = self.embedding_2(x_sketch_hidden)

            out = []
            for x, y in zip(x_sketch_h, x_sketch_hidden):
                out.append(self.layernorm(x + y))
            out, num_stroke_list = pad_sequence(out, batch_first=True), batch['num_stroke_per_positive']

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
            x_sketch_hidden = x_sketch_hidden.permute(1, 0, 2).reshape(x_sketch_hidden.shape[1], -1)
            x_sketch_hidden = self.embedding_2(x_sketch_hidden)

            out = []
            for x, y in zip(x_sketch_h, x_sketch_hidden):
                out.append(self.layernorm(x + y))
            out, num_stroke_list = pad_sequence(out, batch_first=True), batch['num_stroke_per_negative']

        return out, num_stroke_list



class VGG_Network(nn.Module):
    def __init__(self, hp):
        super(VGG_Network, self).__init__()
        backbone = backbone_.vgg16(pretrained=True) #vgg16, vgg19_bn

        self.features = backbone.features
        self.avgpool = backbone.avgpool
        backbone.classifier._modules['6'] = nn.Linear(4096, 250)
        self.classifier = backbone.classifier

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x



class Resnet_Network(nn.Module):
    def __init__(self, hp):
        super(Resnet_Network, self).__init__()
        backbone = backbone_.resnet50(pretrained=True) #resnet50, resnet18, resnet34

        self.features = nn.Sequential()
        for name, module in backbone.named_children():
            if name not in ['avgpool', 'fc']:
                self.features.add_module(name, module)

        self.pool_method =  nn.AdaptiveMaxPool2d(1) # as default

        if hp.dataset_name == 'TUBerlin':
            num_class = 250
        else:
            num_class = 125

        self.classifier = nn.Linear(2048, num_class)

    def forward(self, input, bb_box = None):
        x = self.features(input)
        x = self.pool_method(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
