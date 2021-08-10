import torch.nn as nn
from network import VGG_Network, InceptionV3_Network, Resnet50_Network
from torch import optim
import torch
import time
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image
from tqdm import tqdm
from tensorplot import Visualizer
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

# metric imports
from scipy.spatial.distance import cdist
from utils import compressITQ
from utils import apsak, precak, eval_redraw

import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FGSBIR_Model(nn.Module):
    def __init__(self, hp):
        super(FGSBIR_Model, self).__init__()
        self.sample_embedding_network = eval(hp.backbone_name + "_Network(hp)")
        self.loss = nn.TripletMarginLoss(margin=0.2)
        self.sample_train_params = self.parameters()
        self.optimizer = optim.Adam(self.sample_train_params, hp.learning_rate)
        self.hp = hp
        self.step = 0
        # self.Vis = Visualizer(hp.saved_models)
    
    def train_model(self, batch):
        self.step += 1
        self.train()
        self.optimizer.zero_grad()

        positive_feature = self.sample_embedding_network(batch["positive_img"].to(device))
        negative_feature = self.sample_embedding_network(batch["negative_img"].to(device))
        sample_feature = self.sample_embedding_network(batch["sketch_img"].to(device))
        target_coord = batch['sketch_five_point'].float().to(device)

        target_coord[:, :, :2] = target_coord[:, :, :2] / 256.0
        triplet_loss = self.loss(sample_feature, positive_feature, negative_feature)

        triplet_loss.backward()
        self.optimizer.step()

        return triplet_loss.item()

    def evaluate(self, datloader_Test, datloader_Test_photo):
        print('\nEvaluation:')
        start_time = time.time()
        Image_Feature_ALL = []
        Image_Name = []
        Sketch_Feature_ALL = []
        Sketch_Name = []
        Image_path = []        
        self.eval()

        for i_batch, batch in enumerate(tqdm(datloader_Test_photo, desc='Extract photo', disable=self.hp.disable_tqdm)):
            positive_feature = self.sample_embedding_network(batch["positive_img"].to(device)).cpu()
            Image_Name.extend(batch["positive_label"].numpy())
            Image_Feature_ALL.extend(positive_feature.numpy())
            Image_path.extend(batch['path'])

        for i_batch, batch in enumerate(tqdm(datloader_Test, desc='Extract sketch', disable=self.hp.disable_tqdm)):
            sketch_feature = self.sample_embedding_network(batch["sketch_img"].to(device)).cpu()
            Sketch_Feature_ALL.extend(sketch_feature.numpy())
            Sketch_Name.extend(batch["sketch_label"].numpy())

        print("Time taken to accumulate all the features {0:.4f}".format(time.time() - start_time))

        # Compute mAP
        print("Computing evaluation metrics...", end="")
        # pdb.set_trace()
        # Compute similarity
        t = time.time()
        sim_euc = np.exp(-cdist(Sketch_Feature_ALL, Image_Feature_ALL, metric="euclidean"))
        time_euc = (time.time() - t) / len(Sketch_Name)
        print("Time taken to calculate euclidean similarity {0:.4f}".format(time_euc))

        str_sim = (np.expand_dims(Sketch_Name, axis=1) == np.expand_dims(Image_Name, axis=0)) * 1

        apsall = apsak(sim_euc, str_sim)
        aps200 = apsak(sim_euc, str_sim, k=200)
        prec100, _ = precak(sim_euc, str_sim, k=100)
        prec200, _ = precak(sim_euc, str_sim, k=200)

        valid_data = {
            "aps@all": apsall,
            "aps@200": aps200,
            "prec@100": prec100,
            "prec@200": prec200,
            "sim_euc": sim_euc,
            "time_euc": time_euc,
            "str_sim": str_sim,
        }

        return valid_data
