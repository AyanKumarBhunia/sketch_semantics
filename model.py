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

        pos_score = self.score(corr_xpos, num_stroke_anc, num_stroke_pos)
        neg_score = self.score(corr_xneg, num_stroke_anc, num_stroke_neg)

        loss_ncn = neg_score - pos_score
        return loss_ncn, corr_xpos

    def train_model(self, batch):
        self.train()
        self.optimizer.zero_grad()
        loss, _ = self.calc_loss(batch)  # neg_score - pos_score
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate_old(self, dataloader_Test):
        self.eval()
        test_loss = 0
        start_time = time.time()
        for i_batch, batch in enumerate(tqdm(dataloader_Test, desc='Testing', disable=self.hp.disable_tqdm)):
            test_loss, _ = self.calc_loss(batch)

        test_loss /= len(dataloader_Test.dataset)
        print(f'\nTest set: Average loss: {test_loss:.5f}, Time_Takes: {(time.time() - start_time):.5f}\n')

        return test_loss

    def evaluate(self, dataloader_Test):    # in Progress
        self.eval()
        accuracy = 0
        start_time = time.time()
        for i_batch, batch in enumerate(tqdm(dataloader_Test, desc='Testing', disable=self.hp.disable_tqdm)):
            _, corr_pos = self.calc_loss(batch)
            # anc to positive matching
            anc_max = torch.argmax(corr_pos, dim=2)  # argmax of N2 w.r.t to N1  -- index tensor  b x N1
            pos_max = torch.argmax(corr_pos, dim=1)  # argmax of N1 w.r.t to N2  -- index tensor  b x N2

            anc_index = torch.arange(anc_max.shape[1], device=device).unsqueeze(0).repeat(anc_max.shape[0], 1)  # b x N1
            accuracy += (anc_index == torch.gather(pos_max, 1, anc_max)).sum()


            # for i_sample, sample in enumerate(anc_max):
            #     for i_val, val in enumerate(sample):
            #         result.append(float(pos_max[i_sample, val] == i_val))
            accuracy /= len(dataloader_Test.dataset)
            print(f'\nTest set: Average loss: {accuracy*100:.5f}, Time_Takes: {(time.time() - start_time):.5f}\n')

        # rank = torch.zeros(len(Sketch_Name))
        # Image_Feature_ALL = torch.stack(Image_Feature_ALL)
        #
        # for num, sketch_feature in enumerate(Sketch_Feature_ALL):
        #     s_name = Sketch_Name[num]
        #     sketch_query_name = '_'.join(s_name.split('/')[-1].split('_')[:-1])
        #     position_query = Image_Name.index(sketch_query_name)
        #
        #     distance = F.pairwise_distance(sketch_feature.unsqueeze(0), Image_Feature_ALL)
        #     target_distance = F.pairwise_distance(sketch_feature.unsqueeze(0),
        #                                           Image_Feature_ALL[position_query].unsqueeze(0))
        #
        #     rank[num] = distance.le(target_distance).sum()
        #
        # top1 = rank.le(1).sum().numpy() / rank.shape[0]
        # top10 = rank.le(10).sum().numpy() / rank.shape[0]

        return accuracy
