import pdb
import numpy as np
from PIL import Image, ImageDraw
from rasterize import rasterize_Sketch
from network import *
from torch import optim
import torch
import time
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_mask(num_strokes, dim=512):
    max_len = max(num_strokes)
    mask = torch.zeros((len(num_strokes), max_len))

    for i, n_stroke in enumerate(num_strokes):
        mask[i, :n_stroke] = torch.ones((n_stroke,))
    return mask.unsqueeze(-1).expand(-1, -1, dim).to(device)


class Sketch_Classification(nn.Module):
    def __init__(self, hp):
        super(Sketch_Classification, self).__init__()
        self.Network = Stroke_Embedding_Network(hp)
        self.train_params = self.parameters()
        self.optimizer = optim.Adam(self.train_params, hp.learning_rate)
        self.loss = loss
        self.CE_loss = nn.CrossEntropyLoss()
        self.hp = hp
        self.neighbour = NeighbourhoodConsensus2D(use_conv=hp.use_conv, pool=hp.pool, k_size=hp.k_size)

    def train_model(self, batch):
        self.train()
        self.optimizer.zero_grad()

        output_anc, num_stroke_anc = self.Network(batch, type='anchor')
        output_pos, num_stroke_pos = self.Network(batch, type='positive')
        output_neg, num_stroke_neg = self.Network(batch, type='negative')

        mask_anc, mask_pos, mask_neg = map(make_mask, [num_stroke_anc, num_stroke_pos, num_stroke_neg])

        corr_xpos = self.neighbour(output_anc, output_pos, mask_anc, mask_pos)
        corr_xneg = self.neighbour(output_anc, output_neg, mask_anc, mask_neg)

        loss_ncn = self.loss(corr_xpos, corr_xneg)

        # Creating classification accuracy metric
        output_CE = torch.stack([sample.sum(dim=0) for sample in output_anc])
        output_CE = self.Network.classifier(output_CE)
        label_tensor = torch.LongTensor(batch['label']).to(device)
        loss_ce = self.CE_loss(output_CE, label_tensor)

        (loss_ncn + 0.05*loss_ce).backward()

        self.optimizer.step()
        return loss_ncn.item(), loss_ce.item()

    def evaluate(self, dataloader_Test):
        self.eval()
        correct = 0
        test_loss = 0
        start_time = time.time()
        for i_batch, batch in enumerate(tqdm(dataloader_Test, desc='Testing', disable=self.hp.disable_tqdm)):

            output_raw, _ = self.Network(batch, type='anchor')
            output_CE = torch.stack([sample.sum(dim=0) for sample in output_raw])
            output_CE = self.Network.classifier(output_CE)
            label_tensor = torch.LongTensor(batch['label'])

            test_loss += self.CE_loss(output_CE, label_tensor.to(device)).item()
            prediction = output_CE.argmax(dim=1, keepdim=True).to('cpu')
            correct += prediction.eq(label_tensor.view_as(prediction)).sum().item()

        test_loss /= len(dataloader_Test.dataset)
        accuracy = 100. * correct / len(dataloader_Test.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Time_Takes: {}\n'.format(
            test_loss, correct, len(dataloader_Test.dataset), accuracy, (time.time() - start_time)))

        return accuracy
