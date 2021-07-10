from NCNet import *
from network import Stroke_Embedding_Network
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
        self.score = corr_score
        self.hp = hp
        self.neighbour = NeighbourhoodConsensus2D(use_conv=hp.use_conv, pool=hp.pool, k_size=hp.k_size)

    def calc_loss(self, batch):
        output_anc, num_stroke_anc = self.Network(batch, type='anchor')
        output_pos, num_stroke_pos = self.Network(batch, type='positive')
        output_neg, num_stroke_neg = self.Network(batch, type='negative')

        mask_anc, mask_pos, mask_neg = map(make_mask, [num_stroke_anc, num_stroke_pos, num_stroke_neg])

        corr_xpos = self.neighbour(output_anc, output_pos, mask_anc, mask_pos)
        corr_xneg = self.neighbour(output_anc, output_neg, mask_anc, mask_neg)

        pos_score = self.score(corr_xpos)
        neg_score = self.score(corr_xneg)

        loss_ncn = neg_score - pos_score
        return loss_ncn

    def train_model(self, batch):
        self.train()
        self.optimizer.zero_grad()
        loss = self.calc_loss(batch) # neg_score - pos_score
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, dataloader_Test):
        self.eval()
        test_loss = 0
        start_time = time.time()
        for i_batch, batch in enumerate(tqdm(dataloader_Test, desc='Testing', disable=self.hp.disable_tqdm)):
            test_loss += self.calc_loss(batch).item()

        test_loss /= len(dataloader_Test.dataset)
        print(f'\nTest set: Average loss: {test_loss:.5f}, Time_Takes: {(time.time() - start_time):.5f}\n')

        return test_loss
