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
import copy
from torch.nn.utils.rnn import pad_sequence

unseen_classes = ['bat', 'cabin', 'cow', 'dolphin', 'door', 'giraffe', 'helicopter', 'mouse', 'pear', 'raccoon',
                  'rhinoceros', 'saw', 'scissors', 'seagull', 'skyscraper', 'songbird', 'sword', 'tree',
                  'wheelchair', 'windmill', 'window']

class Sketchy_Dataset(data.Dataset):
    def __init__(self, hp, mode = 'Train'):

        self.hp = hp
        self.mode = mode
        self.training = copy.deepcopy(hp.training)
        with open('sketchy_all.pickle', 'rb') as fp:
            train_sketch, test_sketch, self.negetiveSampleDict, self.Coordinate = pickle.load(fp)


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
            sketch_positive_path = random.choice(self.seen_dict[class_name])
            sketch_positive_vector = self.Coordinate[sketch_positive_path]
            sketch_positive = Image.fromarray(rasterize_Sketch(sketch_positive_vector)).convert('RGB')

            possible_negative_class = random.choice(list(set(self.Seen_Class) - set(class_name)))
            sketch_negative_path = random.choice(self.seen_dict[possible_negative_class])
            sketch_negative_vector = self.Coordinate[sketch_negative_path]
            sketch_negative = Image.fromarray(rasterize_Sketch(sketch_negative_vector)).convert('RGB')


        if self.hp.data_encoding_type == '3point':
            stroke_wise_split_anchor = np.split(anchor_sketch_vector,
                                                np.where(anchor_sketch_vector[:,2])[0] + 1, axis=0)[:-1]
            stroke_wise_split_positive = np.split(sketch_positive_vector,
                                                np.where(sketch_positive_vector[:,2])[0] + 1, axis=0)[:-1]
            stroke_wise_split_negative = np.split(sketch_negative_vector,
                                                np.where(sketch_negative_vector[:,2])[0] + 1, axis=0)[:-1]

        elif self.hp.data_encoding_type == '5point':
            anchor_sketch_vector = self.to_delXY(anchor_sketch_vector)
            stroke_wise_split_anchor = np.split(anchor_sketch_vector, np.where(anchor_sketch_vector[:,3])[0] + 1, axis=0)

            sketch_positive_vector = self.to_delXY(sketch_positive_vector)
            stroke_wise_split_positive = np.split(sketch_positive_vector, np.where(sketch_positive_vector[:, 3])[0] + 1,
                                                axis=0)

            sketch_negative_vector = self.to_delXY(sketch_negative_vector)
            stroke_wise_split_negative = np.split(sketch_negative_vector, np.where(sketch_negative_vector[:, 3])[0] + 1,
                                                axis=0)

        else:
            raise ValueError('invalid option for --data_encoding_type. Valid options: 3point/5point')

        stroke_wise_split_anchor = [torch.tensor(x) for x in stroke_wise_split_anchor]
        every_stroke_len_anchor = [len(stroke) for stroke in stroke_wise_split_anchor]
        num_stroke_per_anchor= len(every_stroke_len_anchor)
        assert sum(every_stroke_len_anchor) == anchor_sketch_vector.shape[0]

        stroke_wise_split_positive = [torch.tensor(x) for x in stroke_wise_split_positive]
        every_stroke_len_positive = [len(stroke) for stroke in stroke_wise_split_positive]
        num_stroke_per_positive = len(every_stroke_len_positive)
        assert sum(every_stroke_len_positive) == sketch_positive_vector.shape[0]


        stroke_wise_split_negative = [torch.tensor(x) for x in stroke_wise_split_negative]
        every_stroke_len_negative = [len(stroke) for stroke in stroke_wise_split_negative]
        num_stroke_per_negative = len(every_stroke_len_negative)
        assert sum(every_stroke_len_negative) == sketch_negative_vector.shape[0]


        sample = { 'path': path,
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

        return  sample

    def to_delXY(self, seq):
       len_seq, _ = seq.shape
       new_seq = np.zeros((len_seq, 5), dtype=seq.dtype)
       new_seq[:,:2] = seq[:,:2]
       new_seq[:,3] = seq[:,2]
       new_seq[:,2] = 1-seq[:,2]
       new_seq[len_seq-1,2:] = [0,0,1]
       new_seq[:-1,:2] = new_seq[1:,:2] - new_seq[:-1,:2]
       new_seq[:-1,2:] = new_seq[1:,2:]
       return new_seq[:-1,:]

    def __len__(self):
        if self.mode == 'Train':
            return len(self.Train_Sketch)
        elif self.mode == 'Test':
            return len(self.Test_Sketch)

def collate_self(batch):
    batch_mod = {'path': [], 'label': [],

                 'anchor_sketch_image': [], 'anchor_sketch_vector': [], 'num_stroke_per_anchor': [],
                'every_stroke_len_anchor': [], 'stroke_wise_split_anchor': [],

                 'sketch_positive': [], 'sketch_positive_vector': [], 'num_stroke_per_positive': [],
                 'every_stroke_len_positive': [], 'stroke_wise_split_positive': [],

                 'sketch_negative': [], 'sketch_negative_vector': [], 'num_stroke_per_negative': [],
                 'every_stroke_len_negative': [], 'stroke_wise_split_negative': []
                 }



    for i_batch in batch:

        batch_mod['path'].append(i_batch['path'])
        batch_mod['label'].append(i_batch['label']) # later

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


    batch_mod['anchor_sketch_image'] = torch.stack(batch_mod['anchor_sketch_image'] )
    batch_mod['anchor_sketch_vector'] = pad_sequence(batch_mod['anchor_sketch_vector'], batch_first=True)
    batch_mod['every_stroke_len_anchor'] = torch.tensor(batch_mod['every_stroke_len_anchor'])
    batch_mod['stroke_wise_split_anchor'] = pad_sequence(batch_mod['stroke_wise_split_anchor'], batch_first=True)


    batch_mod['sketch_positive'] = torch.stack(batch_mod['sketch_positive'] )
    batch_mod['sketch_positive_vector'] = pad_sequence(batch_mod['sketch_positive_vector'], batch_first=True)
    batch_mod['every_stroke_len_positive'] = torch.tensor(batch_mod['every_stroke_len_positive'])
    batch_mod['stroke_wise_split_positive'] = pad_sequence(batch_mod['stroke_wise_split_positive'], batch_first=True)


    batch_mod['sketch_negative'] = torch.stack(batch_mod['sketch_negative'])
    batch_mod['sketch_negative_vector'] = pad_sequence(batch_mod['sketch_negative_vector'], batch_first=True)
    batch_mod['every_stroke_len_negative'] = torch.tensor(batch_mod['every_stroke_len_negative'])
    batch_mod['stroke_wise_split_negative'] = pad_sequence(batch_mod['stroke_wise_split_negative'], batch_first=True)

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

    dataset_Train  = Sketchy_Dataset(hp, mode = 'Train')
    dataset_Test = Sketchy_Dataset(hp, mode='Test')

    #dataset_Test.Test_Sketch = dataset_Train.Test_Sketch
    #dataset_Train.Test_Sketch = []
    #dataset_Test.Train_Sketch = []

    dataloader_Train = data.DataLoader(dataset_Train, batch_size=hp.batchsize, shuffle=True,
                                         num_workers=int(hp.nThreads), collate_fn=collate_self)

    dataloader_Test = data.DataLoader(dataset_Test, batch_size=hp.batchsize, shuffle=False,
                                         num_workers=int(hp.nThreads), collate_fn=collate_self)

    return dataloader_Train, dataloader_Test




























# unseen_classes = ['bat', 'cabin', 'cow', 'dolphin', 'door', 'giraffe', 'helicopter', 'mouse', 'pear', 'raccoon',
#                   'rhinoceros', 'saw', 'scissors', 'seagull', 'skyscraper', 'songbird', 'sword', 'tree', 'wheelchair', 'windmill', 'window']
#
# import random
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#
# class FGSBIR_Dataset(data.Dataset):
#     def __init__(self, hp, mode):
#
#         with open('/home/aman/Desktop/VIRUS/Datasets/Sketchy_dataset/sketchy_all.pickle', 'rb') as fp:
#             train_sketch, test_sketch, self.negetiveSampleDict, self.Coordinate = pickle.load(fp)
#
#         #train_set = [x for x in train_sketch if x.split('/')[0] not in unseen_classes]
#         #test_set = [x for x in test_sketch if x.split('/')[0] not in unseen_classes]
#         #self.Train_Sketch = train_set + test_set
#
#         #class_keys = [x for x in self.negetiveSampleDict.keys() if x.split('/')[0] not in unseen_classes]
#
#         eval_train_set = [x for x in train_sketch if x.split('/')[0] in unseen_classes]
#         eval_test_set = [x for x in test_sketch if x.split('/')[0] in unseen_classes]
#
#         self.Test_Sketch = eval_train_set + eval_test_set
#
#         key = [x for x in unseen_classes]
#         value = [x for x in range(len(unseen_classes))]
#         self.string2label = {}
#         self.label2string = {}
#         for i in range(len(key)):
#             self.string2label[key[i]] = value[i]
#             self.label2string[value[i]] = key[i]
#
#
#         transform_list =[transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]
#         transform_list_sketch = [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
#         self.transform = transforms.Compose(transform_list)
#         self.transform_sketch=transforms.Compose(transform_list_sketch)
#
#         self.mode = mode
#         self.root_dir = hp.root_dir
#
#     def __getitem__(self, item):
#         if self.mode == 'Train':
#             sketch_path = self.Test_Sketch[item]
#             positive_sample = self.Test_Sketch[item].split('/')[0] + '/' + \
#                               self.Test_Sketch[item].split('/')[1].split('-')[0]
#             positive_path = os.path.join(self.root_dir, 'photo', positive_sample + '.jpg')
#
#             class_name = self.Test_Sketch[item].split('/')[0]
#
#             #vector_x = self.Coordinate[sketch_path]
#             #sketch_img = rasterize_Sketch(vector_x)
#             #sketch_img = Image.fromarray(sketch_img).convert('RGB')
#
#             edge_map = get_edgemap(positive_path)
#             #positive_img = Image.open(positive_path).convert('RGB')
#
#
#             edge_map = self.transform_sketch(edge_map)
#
#             label = self.string2label[class_name]
#             return edge_map, label
#
#         elif self.mode == 'Test':
#             sketch_path = self.Test_Sketch[item]
#
#             class_name = self.Test_Sketch[item].split('/')[0]
#
#             vector_x = self.Coordinate[sketch_path]
#             sketch_img = rasterize_Sketch(vector_x)
#             sketch_img = Image.fromarray(sketch_img).convert('RGB')
#             sketch_img = self.transform_sketch(sketch_img)
#             label = self.string2label[class_name]
#             return sketch_img, label
#     def __len__(self):
#         return len(self.Test_Sketch)
#
#
#
# def get_edgemap(dir):
#     img = cv2.imread(dir)
#     edges = cv2.Canny(img,100,200)
#     edges = cv2.resize(edges, (256, 256))
#     edges = cv2.GaussianBlur(edges, (1, 1), cv2.BORDER_DEFAULT)
#     edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
#     return edges
#
#
# def get_dataloader(hp):
#
#     dataset_Train  = FGSBIR_Dataset(hp, mode = 'Train')
#     dataloader_Train = data.DataLoader(dataset_Train, batch_size=hp.batchsize, shuffle=True,
#                                          num_workers=int(hp.nThreads))
#
#     dataset_Test  = FGSBIR_Dataset(hp, mode = 'Test')
#     dataloader_Test = data.DataLoader(dataset_Test, batch_size=1, shuffle=False,
#                                          num_workers=int(hp.nThreads))
#
#     return dataloader_Train, dataloader_Test
#
#
#
#
