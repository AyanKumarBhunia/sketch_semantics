'''
# dataset_Train = Dataset_loader(hp, mode="Train")
# dataset_Test = Dataset_loader(hp, mode="Test")
# dataset_Test_photo = Dataset_loader(hp, mode="Test_photo")
#
# dataloader_Train = data.DataLoader(
#     dataset_Train,
#     batch_size=hp.batchsize,
#     shuffle=True,
#     num_workers=int(hp.nThreads),
# )
#
# dataloader_Test = data.DataLoader(
#     dataset_Test,
#     batch_size=hp.batchsize,
#     shuffle=False,
#     num_workers=int(hp.nThreads),
# )
#
# dataloader_Test_photo = data.DataLoader(
#     dataset_Test_photo,
#     batch_size=hp.batchsize,
#     shuffle=False,
#     num_workers=int(hp.nThreads),
# )

# dataloader_Test = data.DataLoader(
#     dataset_Test,
#     batch_size=1,
#     shuffle=False,
#     num_workers=int(hp.nThreads),
# )

# dataloader_Test_photo = data.DataLoader(
#     dataset_Test_photo,
#     batch_size=1,
#     shuffle=False,
#     num_workers=int(hp.nThreads),
# )

# return dataloader_Train, dataloader_Test ,dataloader_Test_photo


# def collate_self(batch):
#     batch_mod = {"img": [], "sketch_boxes": [], "label": [], "sketch_path": []}
#     for i_batch in batch:
#         batch_mod["img"].append(i_batch["img"])
#         batch_mod["sketch_boxes"].append(torch.tensor(i_batch["sketch_boxes"]).float())
#         batch_mod["label"].append(i_batch["label"])
#         batch_mod["sketch_path"].append(i_batch["sketch_path"])
#
#     batch_mod["img"] = torch.stack(batch_mod["img"], dim=0)
#     batch_mod["label"] = torch.tensor(batch_mod["label"])
#
#     return batch_mod


# for key in temp_keys:
#     label = key.split('_')[0]
#     if ' ' in label:                # alarm clock -> alarm_clock
#         label = label.replace(' ', '_')
#     if label not in get_all_classes:
#         rest =  key[key.find('_'):]
#         self.Train_keys.append(label+rest)
#         self.Train_keys.remove(key)      # either way the key will be removed

# temp_keys = self.Test_keys.copy()
# for key in temp_keys:
#     label = key.split('_')[0]
#     if label not in get_all_classes:
#         rest =  key[key.find('_'):]
#         if ' ' in label:                # alarm clock -> alarm_clock
#             label = label.replace(' ', '_')
#             self.Test_keys.append(label+rest)
#         self.Test_keys.remove(key)      # either way the key will be removed

# get unique classes:
# get_all_classes, all_samples = [], []
# for x in self.Train_keys:
#     all_samples.append(x)
#     get_all_classes.append(x[:x.rfind('_')])
# get_all_classes = sorted(list(set(get_all_classes)))     # why are we sorting this though?
'''