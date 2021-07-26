'''
def evaluate(self, dataloader_Test):
    self.eval()
    correct = 0
    test_loss = 0
    start_time = time.time()
    for i_batch, batch in enumerate(tqdm(dataloader_Test, desc='Testing', disable=self.hp.disable_tqdm)):
        # output_anc, num_stroke_anc = self.Network(batch, type='anchor')
        # output_pos, num_stroke_pos = self.Network(batch, type='positive')
        # mask_anc, mask_pos = map(make_mask, [num_stroke_anc, num_stroke_pos])
        # corr_xpos = self.neighbour(output_anc, output_pos, mask_anc, mask_pos)

        output_raw, _ = self.Network(batch, type='anchor')
        output_CE = torch.stack([sample.sum(dim=0) for sample in output_raw])
        output_CE = self.Network.classifier(output_CE)
        label_tensor = torch.LongTensor(batch['label'])

        test_loss += self.CE_loss(output_CE, label_tensor.to(device)).item()
        prediction = output_CE.argmax(dim=1, keepdim=True).to('cpu')
        correct += prediction.eq(label_tensor.view_as(prediction)).sum().item()


        # Creating classification accuracy metric
        # output_CE = torch.stack([sample.sum(dim=0) for sample in output_anc])
        # output_CE = self.Network.classifier(output_CE)
        # label_tensor = torch.LongTensor(batch['label']).to(device)
        # loss_ce = self.CE_loss(output_CE, label_tensor)
        # (loss_ncn + 0.05*loss_ce).backward()

        # with torch.autograd.detect_anomaly():



# DATASET.PY
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
#             train_sketch, test_sketch, self.negativeSampleDict, self.Coordinate = pickle.load(fp)
#
#         #train_set = [x for x in train_sketch if x.split('/')[0] not in unseen_classes]
#         #test_set = [x for x in test_sketch if x.split('/')[0] not in unseen_classes]
#         #self.Train_Sketch = train_set + test_set
#
#         #class_keys = [x for x in self.negativeSampleDict.keys() if x.split('/')[0] not in unseen_classes]
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




def corr_score(corr_xy):
    corr_B = corr_xy  # .permute(0, 2, 1)
    corr_A = corr_xy.permute(0, 2, 1)

    corr_A.softmax(dim=-1)

    score_A, _ = corr_A.softmax(dim=-1).max(-1)   # (b, N1, N2).max(2) -> (b, N1, 1)
    score_B, _ = corr_B.softmax(dim=-1).max(-1)   # (b, N2, N1).max(2) -> (b, N2, 1)


    print(torch.nansum(score_A), torch.nansum(score_B))
    # score_A_mean = torch.nansum(score_A)/ (1 - torch.isnan(score_A).float()).sum()
    # score_B_mean = torch.nansum(score_B) / (1 - torch.isnan(score_B).float()).sum()


    score_A_mean = (torch.where(torch.isnan(score_A), torch.tensor(0.0).to(device), score_A)).sum() / (1 - torch.isnan(score_A).float()).sum()
    score_B_mean = (torch.where(torch.isnan(score_B), torch.tensor(0.0).to(device), score_B)).sum() / (1 - torch.isnan(score_B).float()).sum()

    # https://discuss.pytorch.org/t/torch-nansum-yields-nan-gradients-unexpectedly/117115/3

    print(score_A_mean, score_B_mean)
    score = (score_A_mean + score_B_mean) / 2

    print(score)
    return score



'''