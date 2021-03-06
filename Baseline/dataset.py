import copy
import torch
import pickle
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import cv2
from random import randint
from PIL import Image
import random
import torchvision.transforms.functional as F
from rasterize import rasterize_Sketch
import numpy as np
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

unseen_classes = ['bat', 'cabin', 'cow', 'dolphin', 'door', 'giraffe', 'helicopter', 'mouse', 'pear', 'raccoon', 'rhinoceros', 'saw', 'scissors', 'seagull', 'skyscraper', 'songbird', 'sword', 'tree', 'wheelchair', 'windmill', 'window']
# /vol/research/sketchCV/datasets/Sketchy/


class Sketchy_Dataset(data.Dataset):
    def __init__(self, hp, mode='Train'):

        self.hp = hp
        self.mode = mode
        self.training = copy.deepcopy(hp.training)
        with open(hp.base_dir + '/../../../datasets/Sketchy/sketchy_all.pickle', 'rb') as fp:
            train_sketch, test_sketch, self.negativeSampleDict, self.Coordinate = pickle.load(fp)

        set_A = [x for x in train_sketch if x.split('/')[0] not in unseen_classes]
        set_B = [x for x in test_sketch if x.split('/')[0] not in unseen_classes]
        self.Train_Sketch = set_A + set_B

        set_A = [x for x in train_sketch if x.split('/')[0] in unseen_classes]
        set_B = [x for x in test_sketch if x.split('/')[0] in unseen_classes]
        self.Test_Sketch = set_A + set_B

        self.Seen_Class = []
        for x in self.Train_Sketch:
            self.Seen_Class.append(x.split('/')[0])
        self.Seen_Class = list(set(self.Seen_Class))
        self.Seen_Class.sort()

        self.seen_dict = {}
        for x in self.Seen_Class:
            self.seen_dict[x] = []
        for x in self.Train_Sketch:
            self.seen_dict[x.split('/')[0]].append(x)

        self.Unseen_Class = []
        for x in self.Test_Sketch:
            self.Unseen_Class.append(x.split('/')[0])
        self.Unseen_Class = list(set(self.Unseen_Class))
        self.Unseen_Class.sort()

        self.unseen_dict = {}
        for x in self.Unseen_Class:
            self.unseen_dict[x] = []
        for x in self.Test_Sketch:
            self.unseen_dict[x.split('/')[0]].append(x)

        get_all_classes = self.Seen_Class + self.Unseen_Class

        self.num2name, self.name2num = {}, {}
        for num, val in enumerate(get_all_classes):
            self.num2name[num] = val
            self.name2num[val] = num
        self.train_transform = get_transform('Train')
        self.test_transform = get_transform('Test')

        print('Total Training Sample {}'.format(len(self.Train_Sketch)))
        print('Total Testing Sample {}'.format(len(self.Test_Sketch)))

    def __getitem__(self, item):

        if self.mode == 'Train':

            path = self.Train_Sketch[item]
            anchor_sketch_vector = self.Coordinate[path]
            anchor_sketch = Image.fromarray(rasterize_Sketch(anchor_sketch_vector)).convert('RGB')

            class_name = path.split('/')[0]
            sketch_positive_path = random.choice(self.seen_dict[class_name])
            sketch_positive_vector = self.Coordinate[sketch_positive_path]
            sketch_positive = Image.fromarray(rasterize_Sketch(sketch_positive_vector)).convert('RGB')

            possible_negative_class = random.choice(list(set(self.Seen_Class) - set(class_name)))
            sketch_negative_path = random.choice(self.seen_dict[possible_negative_class])
            sketch_negative_vector = self.Coordinate[sketch_negative_path]
            sketch_negative = Image.fromarray(rasterize_Sketch(sketch_negative_vector)).convert('RGB')

        else:
            path = self.Test_Sketch[item]
            anchor_sketch_vector = self.Coordinate[path]
            anchor_sketch = Image.fromarray(rasterize_Sketch(anchor_sketch_vector)).convert('RGB')

            class_name = path.split('/')[0]
            sketch_positive_path = random.choice(self.unseen_dict[class_name])
            sketch_positive_vector = self.Coordinate[sketch_positive_path]
            sketch_positive = Image.fromarray(rasterize_Sketch(sketch_positive_vector)).convert('RGB')

            possible_negative_class = random.choice(list(set(self.Seen_Class) - set(class_name)))
            sketch_negative_path = random.choice(self.seen_dict[possible_negative_class])
            sketch_negative_vector = self.Coordinate[sketch_negative_path]
            sketch_negative = Image.fromarray(rasterize_Sketch(sketch_negative_vector)).convert('RGB')

        if self.hp.data_encoding_type == '3point':
            stroke_wise_split_anchor_list = np.split(anchor_sketch_vector,
                                                     np.where(anchor_sketch_vector[:, 2])[0] + 1, axis=0)[:-1]
            stroke_wise_split_positive_list = np.split(sketch_positive_vector,
                                                       np.where(sketch_positive_vector[:, 2])[0] + 1, axis=0)[:-1]
            stroke_wise_split_negative_list = np.split(sketch_negative_vector,
                                                       np.where(sketch_negative_vector[:, 2])[0] + 1, axis=0)[:-1]

        elif self.hp.data_encoding_type == '5point':
            anchor_sketch_vector = self.to_delXY(anchor_sketch_vector)
            stroke_wise_split_anchor_list = np.split(anchor_sketch_vector,
                                                     np.where(anchor_sketch_vector[:, 3])[0] + 1, axis=0)

            sketch_positive_vector = self.to_delXY(sketch_positive_vector)
            stroke_wise_split_positive_list = np.split(sketch_positive_vector,
                                                       np.where(sketch_positive_vector[:, 3])[0] + 1, axis=0)

            sketch_negative_vector = self.to_delXY(sketch_negative_vector)
            stroke_wise_split_negative_list = np.split(sketch_negative_vector,
                                                       np.where(sketch_negative_vector[:, 3])[0] + 1, axis=0)

        else:
            raise ValueError(
                'invalid option for --data_encoding_type. Valid options: 3point/5point')

        stroke_wise_split_anchor = [torch.from_numpy(x) for x in stroke_wise_split_anchor_list]
        every_stroke_len_anchor = [len(stroke) for stroke in stroke_wise_split_anchor]
        num_stroke_per_anchor = len(every_stroke_len_anchor)
        assert sum(every_stroke_len_anchor) == anchor_sketch_vector.shape[0]

        stroke_wise_split_positive = [torch.from_numpy(x) for x in stroke_wise_split_positive_list]
        every_stroke_len_positive = [len(stroke) for stroke in stroke_wise_split_positive]
        num_stroke_per_positive = len(every_stroke_len_positive)
        assert sum(every_stroke_len_positive) == sketch_positive_vector.shape[0]

        stroke_wise_split_negative = [torch.from_numpy(x) for x in stroke_wise_split_negative_list]
        every_stroke_len_negative = [len(stroke) for stroke in stroke_wise_split_negative]
        num_stroke_per_negative = len(every_stroke_len_negative)
        assert sum(every_stroke_len_negative) == sketch_negative_vector.shape[0]

        sample = {'path': path,
                  'label': self.name2num[class_name],

                  'anchor_sketch_image': self.train_transform(anchor_sketch),
                  'anchor_sketch_vector': anchor_sketch_vector,
                  'num_stroke_per_anchor': num_stroke_per_anchor,
                  'every_stroke_len_anchor': every_stroke_len_anchor,
                  'stroke_wise_split_anchor': stroke_wise_split_anchor,

                  'sketch_positive': self.train_transform(sketch_positive),
                  'sketch_positive_vector': sketch_positive_vector,
                  'num_stroke_per_positive': num_stroke_per_positive,
                  'every_stroke_len_positive': every_stroke_len_positive,
                  'stroke_wise_split_positive': stroke_wise_split_positive,

                  'sketch_negative': self.train_transform(sketch_negative),
                  'sketch_negative_vector': sketch_negative_vector,
                  'num_stroke_per_negative': num_stroke_per_negative,
                  'every_stroke_len_negative': every_stroke_len_negative,
                  'stroke_wise_split_negative': stroke_wise_split_negative
                  }

        return sample

    def to_delXY(self, seq):
        len_seq, _ = seq.shape
        new_seq = np.zeros((len_seq, 5), dtype=seq.dtype)
        new_seq[:, :2] = seq[:, :2]
        new_seq[:, 3] = seq[:, 2]
        new_seq[:, 2] = 1-seq[:, 2]
        new_seq[len_seq-1, 2:] = [0, 0, 1]
        new_seq[:-1, :2] = new_seq[1:, :2] - new_seq[:-1, :2]
        new_seq[:-1, 2:] = new_seq[1:, 2:]
        return new_seq[:-1, :]

    def __len__(self):
        if self.mode == 'Train':
            return len(self.Train_Sketch)
        elif self.mode == 'Test':
            return len(self.Test_Sketch)


def pad_sequence(sequences, batch_first=False, padding_value=0.0):
    # (list[tensor], bool, float) -> tensor

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max([s.size(0) for s in sequences])
    # max_len_ = max([s.size(1) for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].new_full(out_dims, padding_value)
    mask_tensor = torch.zeros((len(sequences), max_len))

    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
            mask_tensor[i, :length] = 1.
        else:
            out_tensor[:length, i, ...] = tensor
            mask_tensor[:length, i] = 1.

    return out_tensor, mask_tensor


def collate_self(batch):
    batch_mod = {'path': [], 'label': [],

                 'anchor_sketch_image': [], 'anchor_sketch_vector': [], 'num_stroke_per_anchor': [],
                 'every_stroke_len_anchor': [], 'stroke_wise_split_anchor': [],  'anchor_stroke_mask': [],

                 'sketch_positive': [], 'sketch_positive_vector': [], 'num_stroke_per_positive': [],
                 'every_stroke_len_positive': [], 'stroke_wise_split_positive': [], 'positive_stroke_mask': [],

                 'sketch_negative': [], 'sketch_negative_vector': [], 'num_stroke_per_negative': [],
                 'every_stroke_len_negative': [], 'stroke_wise_split_negative': [], 'negative_stroke_mask': [],
                 }

    for i_batch in batch:

        batch_mod['path'].append(i_batch['path'])
        batch_mod['label'].append(i_batch['label'])  # later

        batch_mod['anchor_sketch_image'].append(i_batch['anchor_sketch_image'])
        batch_mod['anchor_sketch_vector'].append(torch.tensor(i_batch['anchor_sketch_vector']))
        batch_mod['num_stroke_per_anchor'].append(i_batch['num_stroke_per_anchor'])
        batch_mod['every_stroke_len_anchor'].extend(i_batch['every_stroke_len_anchor'])
        batch_mod['stroke_wise_split_anchor'].extend(i_batch['stroke_wise_split_anchor'])

        batch_mod['sketch_positive'].append(i_batch['sketch_positive'])
        batch_mod['sketch_positive_vector'].append(torch.tensor(i_batch['sketch_positive_vector']))
        batch_mod['num_stroke_per_positive'].append(i_batch['num_stroke_per_positive'])
        batch_mod['every_stroke_len_positive'].extend(i_batch['every_stroke_len_positive'])
        batch_mod['stroke_wise_split_positive'].extend(i_batch['stroke_wise_split_positive'])

        batch_mod['sketch_negative'].append(i_batch['sketch_negative'])
        batch_mod['sketch_negative_vector'].append(torch.tensor(i_batch['sketch_negative_vector']))
        batch_mod['num_stroke_per_negative'].append(i_batch['num_stroke_per_negative'])
        batch_mod['every_stroke_len_negative'].extend(i_batch['every_stroke_len_negative'])
        batch_mod['stroke_wise_split_negative'].extend(i_batch['stroke_wise_split_negative'])

    batch_mod['anchor_sketch_image'] = torch.stack(batch_mod['anchor_sketch_image'])
    batch_mod['anchor_sketch_vector'], batch_mod['anchor_stroke_mask'] = pad_sequence(batch_mod['anchor_sketch_vector'], batch_first=True)
    batch_mod['every_stroke_len_anchor'] = torch.tensor(batch_mod['every_stroke_len_anchor'])
    batch_mod['stroke_wise_split_anchor'], _ = pad_sequence(batch_mod['stroke_wise_split_anchor'], batch_first=True)

    batch_mod['sketch_positive'] = torch.stack(batch_mod['sketch_positive'])
    batch_mod['sketch_positive_vector'], batch_mod['positive_stroke_mask'] = pad_sequence(batch_mod['sketch_positive_vector'], batch_first=True)
    batch_mod['every_stroke_len_positive'] = torch.tensor(batch_mod['every_stroke_len_positive'])
    batch_mod['stroke_wise_split_positive'], _ = pad_sequence(batch_mod['stroke_wise_split_positive'], batch_first=True)

    batch_mod['sketch_negative'] = torch.stack(batch_mod['sketch_negative'])
    batch_mod['sketch_negative_vector'], batch_mod['negative_stroke_mask'] = pad_sequence(batch_mod['sketch_negative_vector'], batch_first=True)
    batch_mod['every_stroke_len_negative'] = torch.tensor(batch_mod['every_stroke_len_negative'])
    batch_mod['stroke_wise_split_negative'], _ = pad_sequence(batch_mod['stroke_wise_split_negative'], batch_first=True)

    # batch['stroke_wsie_split'] --> [329, 347, 3] --> [no.of stroke, max no.of points in the longest strokes of the batch, 3]
    # batch['anchor_sketch_vector'] = [16, 2277, 3] --> [batchsize, max no.of points a sketch contains in the batch, 3]

    return batch_mod


def get_transform(type):
    transform_list = []
    if type is 'Train':
        transform_list.extend([transforms.Resize(256)])
    elif type is 'Test':
        transform_list.extend([transforms.Resize(256)])
    transform_list.extend(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return transforms.Compose(transform_list)


def get_dataloader(hp):

    dataset_Train = Sketchy_Dataset(hp, mode='Train')
    dataset_Test = Sketchy_Dataset(hp, mode='Test')

    #dataset_Test.Test_Sketch = dataset_Train.Test_Sketch
    #dataset_Train.Test_Sketch = []
    #dataset_Test.Train_Sketch = []

    dataloader_Train = data.DataLoader(dataset_Train, batch_size=hp.batchsize, shuffle=True,
                                       num_workers=int(hp.nThreads), collate_fn=collate_self)

    dataloader_Test = data.DataLoader(dataset_Test, batch_size=hp.batchsize, shuffle=False,
                                      num_workers=int(hp.nThreads), collate_fn=collate_self)

    return dataloader_Train, dataloader_Test