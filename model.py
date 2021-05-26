
from network import *
from torch import optim
import torch
import time
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
        self.hp = hp
        self.neighbour = NeighbourhoodConsensus2D(
            use_conv=hp.use_conv, pool=hp.pool, k_size=hp.k_size)

    def train_model(self, batch):
        self.train()
        self.optimizer.zero_grad()

        mask_x, mask_y, mask_neg = batch['anchor_stroke_mask'], batch[
            'positive_stroke_mask'], batch["negative_stroke_mask"]

        output_x, num_stroke_x = self.Network(batch, type='anchor')
        output_y, num_stroke_y = self.Network(batch, type='positive')
        output_neg, num_stroke_neg = self.Network(batch, type='negative')

        mask_x, mask_y, mask_neg = map(
            make_mask, [num_stroke_x, num_stroke_y, num_stroke_neg])

        corr_xy = self.neighbour(output_x, output_y, mask_x, mask_y)
        corr_xneg = self.neighbour(
            output_x, output_neg, mask_x, mask_neg)

        loss = self.loss(corr_xy, corr_xneg)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, dataloader_Test):
        self.eval()
        correct = 0
        test_loss = 0
        start_time = time.time()
        for i_batch, batch in enumerate(dataloader_Test):

            output = self.Network(batch['image'].to(device))
            test_loss += self.loss(output, batch['label'].to(device)).item()
            prediction = output.argmax(dim=1, keepdim=True).to('cpu')
            correct += prediction.eq(
                batch['label'].view_as(prediction)).sum().item()

        test_loss /= len(dataloader_Test.dataset)
        accuracy = 100. * correct / len(dataloader_Test.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Time_Takes: {}\n'.format(
            test_loss, correct, len(dataloader_Test.dataset), accuracy, (time.time() - start_time)))

        return accuracy
