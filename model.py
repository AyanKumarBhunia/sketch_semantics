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

    def train_model(self, batch):
        self.train()
        self.optimizer.zero_grad()

        output_anc, num_stroke_anc = self.Network(batch, type='anchor')
        output_pos, num_stroke_pos = self.Network(batch, type='positive')
        output_neg, num_stroke_neg = self.Network(batch, type='negative')

        mask_anc, mask_pos, mask_neg = map(make_mask, [num_stroke_anc, num_stroke_pos, num_stroke_neg])

        corr_xpos = self.neighbour(output_anc, output_pos, mask_anc, mask_pos)
        corr_xneg = self.neighbour(output_anc, output_neg, mask_anc, mask_neg)

        pos_score = self.score(corr_xpos, num_stroke_anc, num_stroke_pos)
        neg_score = self.score(corr_xneg, num_stroke_anc, num_stroke_neg)

        loss = neg_score - pos_score
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, dataloader_Test):    # in Progress
        self.eval()
        accuracy = 0
        start_time = time.time()
        for i_batch, batch in enumerate(tqdm(dataloader_Test, desc='Testing', disable=self.hp.disable_tqdm)):

            output_anc, num_stroke_anc = self.Network(batch, type='anchor')
            output_pos, num_stroke_pos = self.Network(batch, type='positive')
            mask_anc, mask_pos = map(make_mask, [num_stroke_anc, num_stroke_pos])
            corr_xpos = self.neighbour(output_anc, output_pos, mask_anc, mask_pos)
            corr_mask = torch.bmm(mask_anc, mask_pos.permute(0, 2, 1)) / 512.0

            anc_max = torch.argmax(corr_xpos, dim=2)  # argmax of N2 w.r.t to N1  -- index tensor  b x N1
            pos_max = torch.argmax(corr_xpos, dim=1)  # argmax of N1 w.r.t to N2  -- index tensor  b x N2

            # anchor to positive matching
            anc_index = torch.arange(anc_max.shape[1], device=device).unsqueeze(0).repeat(anc_max.shape[0], 1)  # b x N1
            acc_anc2pos = (anc_index == torch.gather(pos_max, 1, anc_max)).sum()

            # positive to anchor matching
            pos_index = torch.arange(pos_max.shape[1], device=device).unsqueeze(0).repeat(pos_max.shape[0], 1)  # b x N2
            acc_pos2anc = (pos_index == torch.gather(anc_max, 1, pos_max)).sum()

            # average across tensor is wrong because the entire tensor is not involved for the correlations
            accuracy += (acc_anc2pos + acc_pos2anc)/2

        accuracy /= len(dataloader_Test.dataset)      # taking mean across batch
        print(f'\nTest set: Average loss: {accuracy*100:.5f}, Time_Taken: {(time.time() - start_time):.5f}\n')

        return accuracy