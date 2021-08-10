from torch.nn.utils.rnn import pad_sequence
from utils import *
import torchvision
import numpy as np
import argparse
from rasterize import mydrawPNG_fromlist, rasterize_Sketch
import torchvision.transforms.functional as F
import random
from PIL import Image
import torchvision.transforms as transforms
import torch.utils.data as data
import pickle
import os
import lmdb
import time
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(9001)


class Dataset_Quickdraw(data.Dataset):
    def __init__(self, database, mode='Train',):

        self.hp = database['hp']
        self.Train_keys = database['Train_keys']
        self.Test_keys = database['Test_keys']
        self.Train_photo_keys = database['Train_photo_keys']
        self.Test_photo_keys = database['Test_photo_keys']
        self.name2num = database['name2num']
        self.num2name = database['num2name']
        self.image_path = database['image_path']
        self.label_filter = database['label_filter']
        self.TrainData_ENV = database['TrainData_ENV']
        self.TestData_ENV = database['TestData_ENV']
        self.mode=mode

        print(f'Mode: {mode} | Total Classes: {len(self.Train_photo_keys)} | ', end='')
        if mode !='Test_photo':
            print(f'Number of sketches: {len(eval("self." + mode + "_keys"))} | ', end='')
        if mode == 'Train':
            print(f'Number of images: {sum([len(value) for value in self.Train_photo_keys.values()])}')
        else:
            print(f'Number of images: {len(self.Test_photo_keys)}')

        self.train_transform = get_transform("Train")
        self.test_transform = get_transform("Test")

    def __getitem__(self, item):

        if self.mode == 'Train':
            with self.TrainData_ENV.begin(write=False) as txn:
                sketch_path = self.Train_keys[item]
                sketch_label = sketch_path.split('_')[0]
                if sketch_label in self.label_filter:
                    sketch_label = self.label_filter[sketch_label]
                sample = txn.get(sketch_path.encode("ascii"))
                sketch_points = np.frombuffer(sample).reshape(-1, 3).copy()
                stroke_wise_split_numpy = np.split(sketch_points, np.where(sketch_points[:, 2])[0] + 1, axis=0)[:-1]
                stroke_wise_split = [torch.from_numpy(x) for x in stroke_wise_split_numpy]
                # every_stroke_len = [len(stroke) for stroke in stroke_wise_split]
                num_stroke = len(stroke_wise_split)
                sketch_img = mydrawPNG_fromlist(sketch_points, list(range(num_stroke)))

            positive_sample = random.choice(self.Train_photo_keys[sketch_label])
            negative_labels = list(self.Train_photo_keys.keys())
            negative_labels.remove(sketch_label)
            negative_label = random.choice(negative_labels)
            negative_sample = random.choice(self.Train_photo_keys[negative_label])

            positive_img = Image.open(os.path.join(self.image_path, sketch_label, positive_sample)).convert("RGB")
            negative_img = Image.open(os.path.join(self.image_path, negative_label, negative_sample)).convert("RGB")

            n_flip = random.random()
            if n_flip > 0.5:
                sketch_img = F.hflip(sketch_img)
                positive_img = F.hflip(positive_img)
                negative_img = F.hflip(negative_img)

            sketch_img = self.train_transform(sketch_img)
            positive_img = self.train_transform(positive_img)
            negative_img = self.train_transform(negative_img)

            sample = {'sketch_img': sketch_img,
                        'sketch_label': self.name2num[sketch_label],
                        'sketch_points': sketch_points,
                        "positive_img": positive_img,
                        "positive_label": self.name2num[sketch_label],
                        "negative_img": negative_img,
                        "negative_label": self.name2num[negative_label],
                        # 'num_stroke': num_stroke,
                        # 'sketch_path': sketch_path,
                        # 'every_stroke_len': every_stroke_len,
                        # 'stroke_wise_split': stroke_wise_split,
                        }

        elif self.mode == 'Test':
            with self.TestData_ENV.begin(write=False) as txn:
                sketch_path = self.Test_keys[item]
                sketch_label = sketch_path.split('_')[0]
                if sketch_label in self.label_filter:
                    sketch_label = self.label_filter[sketch_label]
                sample = txn.get(sketch_path.encode())
                sketch_points = np.frombuffer(sample).reshape(-1, 3).copy()
                stroke_wise_split_numpy = np.split(sketch_points, np.where(sketch_points[:, 2])[0] + 1, axis=0)[:-1]
                stroke_wise_split = [torch.from_numpy(x) for x in stroke_wise_split_numpy]
                # every_stroke_len = [len(stroke) for stroke in stroke_wise_split]
                num_stroke = len(stroke_wise_split)
                sketch_img = mydrawPNG_fromlist(sketch_points, list(range(num_stroke)))

            n_flip = random.random()
            if n_flip > 0.5:
                sketch_img = F.hflip(sketch_img)

            sketch_img = self.test_transform(sketch_img)

            sample = {'sketch_img': sketch_img,
                        'sketch_points': sketch_points,
                        'sketch_label': self.name2num[sketch_label],
                        # 'num_stroke': num_stroke,
                        # 'sketch_path': sketch_path,
                        # 'every_stroke_len': every_stroke_len,
                        # 'stroke_wise_split': stroke_wise_split,
                        }

        elif self.mode == "Test_photo":
            path=self.Test_photo_keys[item]
            positive_img = Image.open(os.path.join(self.image_path,path)).convert("RGB")
            positive_image = self.test_transform(positive_img)
            positive_classname=path.split('/')[0]
            positive_label=self.name2num[positive_classname]
            
            sample = {"positive_img": positive_image,
                    "positive_label": positive_label,
                    "path": path
                }
        return sample

    def __len__(self):
        return len(eval(f"self.{self.mode}_keys"))

def get_transform(type):
    
    transform_list = []
    if type == 'Train':
        transform_list.extend([transforms.Resize((240,240)), transforms.CenterCrop(224)])
    else:
        transform_list.extend([transforms.Resize((224,224))])
    # transform_list = []
    # if type == "Train":
    #     transform_list.extend([transforms.Resize((256, 256))])
    # elif type == "Test":
    #     transform_list.extend([transforms.Resize((256, 256))])
    # elif type == "Test_photo":
    #     transform_list.extend([transforms.Resize((256, 256))])
    transform_list.extend(
        [
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(transform_list)


def get_data(hp):    
    data_start = time.time()
    root = os.path.join(hp.base_dir, hp.data_dir, hp.dataset_name)
    image_path = hp.base_dir + '/../../../datasets/QuickDraw/QuickDraw_images_final'
    label_filter = {'alarm clock': 'alarm_clock', 'coffee cup': 'cup'}

    print('Fetching data ...')
    TrainData_ENV = lmdb.open(root+'/QuickDraw_TrainData25', max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
    TestData_ENV = lmdb.open(root+'/QuickDraw_TestData25', max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
    with open(root+"/QuickDraw_Keys25.pickle", "rb") as handle:
        Train_keys, Test_keys = pickle.load(handle)

    print('Pruning samples ...')
    all_classes = os.listdir(image_path) + ['alarm clock', 'coffee cup']
    temp_keys = Train_keys.copy()
    for sample in temp_keys:
        if sample.split('_')[0] not in all_classes:
            Train_keys.remove(sample)

    temp_keys = Test_keys.copy()
    for sample in temp_keys:
        if sample.split('_')[0] not in all_classes:
            Test_keys.remove(sample)
    
    print('Getting unique classes ...')
    get_all_classes = []
    for sample in Train_keys:
        label = sample.split('_')[0]
        if label in label_filter:
            label = label_filter[label]
        get_all_classes.append(label)
    get_all_classes = sorted(list(set(get_all_classes)))

    print('Getting train and test set image-names ...')
    Train_photo_keys, Test_photo_keys = {}, []
    for label in get_all_classes:
        all_images = os.listdir(os.path.join(image_path, label))
        split_index = int(len(all_images) * hp.splitTrain)
        Train_photo_keys[label] = all_images[:split_index]
        Test_photo_keys.extend([label + '/' + sample for sample in all_images[split_index:]])

    print('Mapping labels to dictionary ...')
    num2name, name2num = {}, {}
    for num, val in enumerate(get_all_classes):
        num2name[num] = val
        name2num[val] = num

    database = {'hp': hp,
                'Train_keys': Train_keys,
                'Test_keys': Test_keys,
                'Train_photo_keys': Train_photo_keys,
                'Test_photo_keys': Test_photo_keys,
                'name2num': name2num,
                'num2name': num2name,
                'image_path': image_path,
                'label_filter': label_filter,
                'TrainData_ENV': TrainData_ENV,
                'TestData_ENV': TestData_ENV
                }

    print(f'Done. Time Taken: {time.time()-data_start :.3f} seconds.\n')
    return database


def get_dataloader(database, mode='Train'):
    return data.DataLoader(Dataset_Quickdraw(database, mode), batch_size=database['hp'].batchsize, collate_fn=eval('collate_self_'+mode),
                           shuffle=(mode == 'Train'), num_workers=int(database['hp'].nThreads))
