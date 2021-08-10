import torch
import numpy as np
import multiprocessing
import os
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score
from PIL import Image
from rasterize import mydrawPNG
from torchvision.utils import save_image
import smtplib


def send_email(sender_email, message, bot_email='pinakinathc.bot1@gmail.com', bot_pwd='pinakinathc1995'):
    smtpObj = smtplib.SMTP('smtp.gmail.com', 587) # Connect to google's smtp server
    smtpObj.ehlo()
    smtpObj.starttls()
    smtpObj.login(bot_email, bot_pwd)
    smtpObj.sendmail(bot_email, sender_email, f'Subject: Update from pinakinathc.bot-1\n\n{message}')

def collate_self_Train(batch):
    batch_mod = {'sketch_img': [], 'sketch_label': [],
                #  'stroke_wise_split': [], 'num_stroke': [],
                 'positive_img': [], 'positive_label': [],
                 'negative_img': [], 'negative_label': [],
                 'sketch_five_point': [], 'seq_len': []
                 }
    max_len = max([len(x['sketch_points']) for x in batch])

    for i_batch in batch:
        batch_mod['sketch_img'].append(i_batch['sketch_img'])
        batch_mod['sketch_label'].append(i_batch['sketch_label'])
        batch_mod['positive_img'].append(i_batch['positive_img'])
        batch_mod['positive_label'].append(i_batch['positive_label'])
        batch_mod['negative_img'].append(i_batch['negative_img'])
        batch_mod['negative_label'].append(i_batch['negative_label'])
        # batch_mod['stroke_wise_split'].append(i_batch['stroke_wise_split']) 
        # batch_mod['num_stroke'].append(i_batch['num_stroke']) 
         

        five_point, len_seq = to_Five_Point(i_batch['sketch_points'], max_len)
        # First time step is [0, 0, 0, 0, 0] as start token.
        batch_mod['sketch_five_point'].append(torch.tensor(five_point))
        batch_mod['seq_len'].append(len_seq)

    batch_mod['sketch_img'] = torch.stack(batch_mod['sketch_img'], dim=0)
    batch_mod['positive_img'] = torch.stack(batch_mod['positive_img'], dim=0)
    batch_mod['negative_img'] = torch.stack(batch_mod['negative_img'], dim=0)
    # batch_mod['num_stroke'] = torch.stack(batch_mod['num_stroke'], dim=0)

    batch_mod['sketch_label'] = torch.tensor(batch_mod['sketch_label'])
    batch_mod['positive_label'] = torch.tensor(batch_mod['positive_label'])
    batch_mod['negative_label'] = torch.tensor(batch_mod['negative_label'])
    # batch_mod['stroke_wise_split'] = torch.tensor(batch_mod['stroke_wise_split'])

    batch_mod['sketch_five_point'] = torch.stack(batch_mod['sketch_five_point'], dim=0)
    batch_mod['seq_len'] = torch.tensor(batch_mod['seq_len'])

    return batch_mod


def collate_self_Test(batch):
    batch_mod = {'sketch_img': [], 'sketch_label': [],
                 'positive_label': [], 'positive_img': [],
                 'sketch_five_point': [], 'seq_len': [], 'path': [],}
    max_len = max([len(x['sketch_points']) for x in batch])

    for i_batch in batch:
        batch_mod['sketch_img'].append(i_batch['sketch_img'])
        # batch_mod['positive_img'].append(i_batch['positive_img'])
        batch_mod['sketch_label'].append(i_batch['sketch_label'])   
        # batch_mod['positive_label'].append(i_batch['positive_label'])     
        # batch_mod['path'].append(i_batch['path'])
        # batch_mod['stroke_wise_split'].append(i_batch['stroke_wise_split'])      

        five_point, len_seq = to_Five_Point(i_batch['sketch_points'], max_len)
        # First time step is [0, 0, 0, 0, 0] as start token.
        batch_mod['sketch_five_point'].append(torch.tensor(five_point))
        batch_mod['seq_len'].append(len_seq)

    batch_mod['sketch_img'] = torch.stack(batch_mod['sketch_img'], dim=0)
    # batch_mod['positive_img'] = torch.stack(batch_mod['positive_img'], dim=0)

    batch_mod['sketch_label'] = torch.tensor(batch_mod['sketch_label'])
    # batch_mod['positive_label'] = torch.tensor(batch_mod['positive_label'])
    # batch_mod['stroke_wise_split'] = torch.tensor(batch_mod['stroke_wise_split'])

    batch_mod['sketch_five_point'] = torch.stack(batch_mod['sketch_five_point'], dim=0)
    batch_mod['seq_len'] = torch.tensor(batch_mod['seq_len'])

    return batch_mod


def collate_self_Test_photo(batch):
    batch_mod = {'positive_img': [], 'path': [], 'positive_label': []}

    for i_batch in batch:
        batch_mod['positive_img'].append(i_batch['positive_img'])
        batch_mod['path'].append(i_batch['path'])
        batch_mod['positive_label'].append(i_batch['positive_label'])

    batch_mod['positive_img'] = torch.stack(batch_mod['positive_img'], dim=0)
    batch_mod['positive_label'] = torch.tensor(batch_mod['positive_label'])

    return batch_mod


def to_Five_Point(sketch_points, max_seq_length):
    len_seq = len(sketch_points[:, 0])
    new_seq = np.zeros((max_seq_length, 5))
    new_seq[0:len_seq, :2] = sketch_points[:, :2]
    new_seq[0:len_seq, 3] = sketch_points[:, 2]
    new_seq[0:len_seq, 2] = 1 - new_seq[0:len_seq, 3]
    new_seq[(len_seq - 1):, 4] = 1
    new_seq[(len_seq - 1), 2:4] = 0
    new_seq = np.concatenate((np.zeros((1, 5)), new_seq), axis=0)
    return new_seq, len_seq


def eval_redraw(target, output, seq_len, step, saved_folder, type, num_print=8, side=1):

    batch_redraw = []

    for sample_targ, sample_gen, seq in zip(target[:num_print], output[:num_print], seq_len[:num_print]):

        sample_gen = sample_gen.cpu().numpy()[:seq]
        sample_targ = sample_targ.cpu().numpy()
        sample_targ = to_normal_strokes(sample_targ)
        sample_gen = to_normal_strokes(sample_gen)


        sample_gen[:, :2] = np.round(sample_gen[:, :2] * side)
        image_gen = mydrawPNG(sample_gen)
        image_gen = Image.fromarray(image_gen).convert('RGB')  # PRoblem lies here


        sample_targ[:, :2] = np.round(sample_targ[:, :2] * side)
        image_targ  = mydrawPNG(sample_targ)
        image_targ = Image.fromarray(image_targ).convert('RGB')

        batch_redraw.append(torch.from_numpy(np.array(image_targ)).permute(2, 0, 1))
        batch_redraw.append(torch.from_numpy(np.array(image_gen)).permute(2, 0, 1))

    batch_redraw = torch.stack(batch_redraw).float()
    save_image(batch_redraw, os.path.join(saved_folder, type + '_' + str(step) + '_.jpg'), normalize=True, nrow=2)


def to_normal_strokes(big_stroke):
  """Convert from stroke-5 format (from sketch-rnn paper) back to stroke-3."""
  l = 0
  for i in range(len(big_stroke)):
      if big_stroke[i, 4] > 0:
        l = i
        break
  if l == 0:
      l = len(big_stroke)
  result = np.zeros((l, 3))
  result[:, 0:2] = big_stroke[0:l, 0:2]
  result[:, 2] = big_stroke[0:l, 3]
  result[-1, 2] = 1.
  return result


def ITQ(V, n_iter):
    # Main function for  ITQ which finds a rotation of the PCA embedded data
    # Input:
    #     V: nxc PCA embedded data, n is the number of images and c is the code length
    #     n_iter: max number of iterations, 50 is usually enough
    # Output:
    #     B: nxc binary matrix
    #     R: the ccc rotation matrix found by ITQ
    # Publications:
    #     Yunchao Gong and Svetlana Lazebnik. Iterative Quantization: A
    #     Procrustes Approach to Learning Binary Codes. In CVPR 2011.
    # Initialize with a orthogonal random rotation initialize with a orthogonal random rotation

    bit = V.shape[1]
    np.random.seed(n_iter)
    R = np.random.randn(bit, bit)
    U11, S2, V2 = np.linalg.svd(R)
    R = U11[:, :bit]

    # ITQ to find optimal rotation
    for iter in range(n_iter):
        Z = np.matmul(V, R)
        UX = np.ones((Z.shape[0], Z.shape[1])) * -1
        UX[Z >= 0] = 1
        C = np.matmul(np.transpose(UX), V)
        UB, sigma, UA = np.linalg.svd(C)
        R = np.matmul(UA, np.transpose(UB))

    # Make B binary
    B = UX
    B[B < 0] = 0

    return B, R


def compressITQ(Xtrain, Xtest, n_iter=50):

    # compressITQ runs ITQ
    # Center the data, VERY IMPORTANT
    Xtrain = Xtrain - np.mean(Xtrain, axis=0, keepdims=True)
    Xtest = Xtest - np.mean(Xtest, axis=0, keepdims=True)

    # PCA
    C = np.cov(Xtrain, rowvar=False)
    l, pc = np.linalg.eigh(C, "U")
    idx = l.argsort()[::-1]
    pc = pc[:, idx]
    XXtrain = np.matmul(Xtrain, pc)
    XXtest = np.matmul(Xtest, pc)

    # ITQ
    _, R = ITQ(XXtrain, n_iter)

    Ctrain = np.matmul(XXtrain, R)
    Ctest = np.matmul(XXtest, R)

    Ctrain = Ctrain > 0
    Ctest = Ctest > 0

    return Ctrain, Ctest


def prec(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    if k is not None:
        pr = len(act_set & pred_set) / min(k, len(pred_set))
    else:
        pr = len(act_set & pred_set) / max(len(pred_set), 1)
    return pr


def rec(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    re = len(act_set & pred_set) / max(len(act_set), 1)
    return re


def precak(sim, str_sim, k=None):
    act_lists = [np.nonzero(s)[0] for s in str_sim]
    pred_lists = np.argsort(-sim, axis=1)
    num_cores = min(multiprocessing.cpu_count(), 32)
    nq = len(act_lists)
    preck = Parallel(n_jobs=num_cores)(
        delayed(prec)(act_lists[iq], pred_lists[iq], k) for iq in range(nq)
    )
    reck = Parallel(n_jobs=num_cores)(
        delayed(rec)(act_lists[iq], pred_lists[iq], k) for iq in range(nq)
    )
    return np.mean(preck), np.mean(reck)


def aps(sim, str_sim):
    nq = str_sim.shape[0]
    num_cores = min(multiprocessing.cpu_count(), 32)
    aps = Parallel(n_jobs=num_cores)(
        delayed(average_precision_score)(str_sim[iq], sim[iq]) for iq in range(nq)
    )
    return aps


def apsak(sim, str_sim, k=None):

    # if (sim.size * sim.itemsize)/1024**3 > 7.5:
    #     print('Excessive memory allocation. Exceeds 7.5 GB')
    #     exit(1)
    idx = (-sim).argsort()[:, :k]
    sim_k = np.array([sim[i, id] for i, id in enumerate(idx)])
    str_sim_k = np.array([str_sim[i, id] for i, id in enumerate(idx)])
    idx_nz = np.where(str_sim_k.sum(axis=1) != 0)[0]
    sim_k = sim_k[idx_nz]
    str_sim_k = str_sim_k[idx_nz]
    aps_ = np.zeros((sim.shape[0]), dtype=np.float)
    aps_[idx_nz] = aps(sim_k, str_sim_k)
    return aps_


def stroke_wise_split_5point():
    return None