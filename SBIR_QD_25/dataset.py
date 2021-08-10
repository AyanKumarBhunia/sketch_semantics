import torch
import pickle
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import os
import random
import copy
import cv2
import numpy as np
import pdb
from natsort import natsorted
from glob import glob
from random import randint
from PIL import Image
from rasterize import rasterize_Sketch
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dataset_loader(data.Dataset):
    def __init__(self, hp, mode="Train"):
        self.mode = mode
        get_all_classes=[]
        if hp.dataset_name=="Sketchy":
            coordinate_path = os.path.join(hp.base_dir, hp.data_dir, hp.dataset_name, "sketchy_all.pickle")
            self.photo_path = os.path.join(hp.base_dir, hp.data_dir, hp.dataset_name, "extended_photo_v2/")
            with open(coordinate_path, "rb") as fp:
                (
                    train_sketch,
                    test_sketch,
                    self.negetiveSampleDict,
                    self.Coordinate,
                ) = pickle.load(fp)
            get_all_classes = natsorted(os.listdir(self.photo_path + "/train"))
            
        elif hp.dataset_name=="TUBerlin":
            coordinate_path = os.path.join(hp.base_dir, hp.data_dir, hp.dataset_name, 'TU_Berlin.pickle')
            self.photo_path=hp.base_dir
    
            with open(coordinate_path, 'rb') as fp:
                self.Coordinate = pickle.load(fp)
    
            total_del = 0
            all_keys = list(self.Coordinate.keys())
            for key in all_keys:
                if len(self.Coordinate[key]) > 300:
                    del self.Coordinate[key]
                    total_del += 1
    
            print('Total Number of samples deleted: {}'.format(total_del))
            #get unique classes:
            get_all_classes, all_samples = [], []
            for x in list(self.Coordinate.keys()):
                get_all_classes.append(x.split('/')[0])
                all_samples.append(x)
            get_all_classes = list(set(get_all_classes))
            get_all_classes.sort()
            
            train_sketch=[]
            test_sketch=[]
            for class_name in get_all_classes:
                per_class_data = np.array([x for x in all_samples if class_name == x.split('/')[0]])
                per_class_Train = per_class_data[random.sample(range(len(per_class_data)), int(len(per_class_data) * hp.splitTrain))]
                per_class_Test = set(per_class_data) - set(per_class_Train)
                train_sketch.extend(list(per_class_Train))
                test_sketch.extend(list(per_class_Test))
        
        self.train_images_classes = {}
        self.test_images_classes = {}
        self.num2name, self.name2num = {}, {}
        self.test_images_values = {}
        self.test_photo_all = []
        # pdb.set_trace()

        for index, classname in enumerate(get_all_classes):
            train_images_list = natsorted(glob(os.path.join(self.photo_path, f"train/{classname}/*")))
            test_images_list = natsorted(glob(os.path.join(self.photo_path, f"test/{classname}/*")))
            
            self.num2name[index] = classname
            self.name2num[classname] = index

            self.train_images_classes[classname] = train_images_list
            self.test_images_classes[classname] = test_images_list
            self.test_photo_all.extend(test_images_list)

            if len(self.train_images_classes[classname])<1:
                pdb.set_trace()
                print(classname,len(self.train_images_classes[classname]),
                      os.path.join(self.photo_path, f"train/{class_name}/*"))

        self.Train_List, self.Test_List = train_sketch, test_sketch
        self.data = "coordinate"
        
        self.train_transform = get_transform("Train")
        self.test_transform = get_transform("Test")

        print(f'Total Training Sample {len(self.Train_List)} | '
              f'Total Testing Sample {len(self.Test_List)} | Mode: {self.mode} ')

    def __getitem__(self, item):
        sample = {}

        if self.mode == "Train":
            path = self.Train_List[item]

            if self.data == "coordinate":
                vector_x = self.Coordinate[path]
                sketch_img, sketch_points = rasterize_Sketch(vector_x)
                sketch_image = Image.fromarray(sketch_img).convert("RGB")
                sketch_classname = path.split("/")[0]
                
                positive_path = self.train_images_classes[f"{sketch_classname}"][
                    randint(0, len(self.train_images_classes[f"{sketch_classname}"]) - 1)]
                positive_classname = sketch_classname

                possible_list = list(self.train_images_classes.keys())
                possible_list.remove(sketch_classname)

                negative_classname = possible_list[randint(0, len(possible_list) - 1)]
                negative_path = self.train_images_classes[f"{negative_classname}"][
                    randint(0, len(self.train_images_classes[f"{negative_classname}"]) - 1)]
                # print(path,positive_path,negative_path)
                positive_img = Image.open(positive_path).convert("RGB")
                negative_img = Image.open(negative_path).convert("RGB")
                
                n_flip = random.random()
                if n_flip > 0.5:
                    sketch_image = F.hflip(sketch_image)
                    positive_img = F.hflip(positive_img)
                    negative_img = F.hflip(negative_img)

                sketch_image = self.train_transform(sketch_image)
                positive_img = self.train_transform(positive_img)
                negative_img = self.train_transform(negative_img)

                sample = {
                    "sketch_img": sketch_image,
                    "sketch_label": self.name2num[sketch_classname],
                    "sketch_points": sketch_points,
                    "positive_img": positive_img,
                    "positive_label": self.name2num[positive_classname],
                    "negative_img": negative_img,
                    "negative_label": self.name2num[negative_classname],
                }
                
        elif self.mode == "Test":
            path = self.Test_List[item]
            if self.data == "coordinate":
                vector_x = self.Coordinate[path]
                sketch_img, sketch_points = rasterize_Sketch(vector_x)
                sketch_image = Image.fromarray(sketch_img).convert("RGB")
                sketch_classname = path.split("/")[0]

                positive_path = self.test_images_classes[f"{sketch_classname}"][
                    randint(0, len(self.test_images_classes[f"{sketch_classname}"]) - 1)]
                positive_classname = sketch_classname
                positive_img = Image.open(positive_path).convert("RGB")

                sketch_image = self.test_transform(sketch_image)
                positive_image = self.test_transform(positive_img)

                # sample = {
                #     "sketch_img": sketch_image,
                #     "sketch_label": self.name2num[sketch_classname],
                #     "positive_img": positive_image,
                #     "positive_label": self.name2num[positive_classname],
                #     "test_data":self.test_images_values,
                #     "name2num":self.name2num
                # }
                sample = {
                    "sketch_img": sketch_image,
                    "sketch_label": self.name2num[sketch_classname],
                    "sketch_points": sketch_points,
                    "positive_img": positive_image,
                    }
        
        elif self.mode == "Test_photo":
            path=self.test_photo_all[item]
            positive_img = Image.open(path).convert("RGB")
            positive_image = self.test_transform(positive_img)
            positive_classname=path.split('/')[-2]
            positive_label=self.name2num[positive_classname]
            
            sample = {
                    "positive_img": positive_image,
                    "positive_label": positive_label,
                    "path": path
                }
            
        return sample

    def __len__(self):
        if self.mode == "Train":
            return len(self.Train_List)
        elif self.mode == "Test":
            return len(self.Test_List)
        elif self.mode == "Test_photo":
            return len(self.test_photo_all)


def get_transform(type):
    transform_list = []
    if type == "Train":
        transform_list.extend([transforms.Resize((256, 256))])
    elif type == "Test":
        transform_list.extend([transforms.Resize((256, 256))])
    elif type == "Test_photo":
        transform_list.extend([transforms.Resize((256, 256))])
    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(transform_list)


def get_dataloader(hp, mode='Train'):
    return data.DataLoader(Dataset_loader(hp, mode), batch_size=hp.batchsize, collate_fn=eval('collate_self_'+mode),
                           shuffle=(mode == 'Train'), num_workers=int(hp.nThreads))