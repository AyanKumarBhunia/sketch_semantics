import torch
import os
import numpy as np
from datetime import datetime
from PIL import Image, ImageDraw
from rasterize import rasterize_Sketch
from model import make_mask

time_id = datetime.now().strftime("%b%d_%H-%M-%S")


def show(step, batch, model, save_path):
    if not os.path.isdir(save_path + '/' + time_id):
        os.makedirs(save_path + '/' + time_id)

    output_anc, num_stroke_anc = model.Network(batch, type='anchor')    # b,N1,512
    output_pos, num_stroke_pos = model.Network(batch, type='positive')  # b,N2,512
    mask_anc, mask_pos = map(make_mask, [num_stroke_anc, num_stroke_pos])
    corr_xpos = model.neighbour(output_anc, output_pos, mask_anc, mask_pos)

    '''
    I need the n1 x n2 matrix
    For every stroke in N1, the stroke having highest correlation will be painted.
    '''
    i_sample = 0                      # range = 0 to batch-size
    A = corr_xpos[i_sample]           # Taking 1 sample out of the batch
    anc_max = torch.argmax(A, dim=1)  # 1 x N1

    anc_vec = batch['anchor_sketch_vector']
    anc_img_orig = Image.fromarray(255 - rasterize_Sketch(anc_vec[i_sample].numpy())).convert('RGB')
    anc_stroke_num = batch['num_stroke_per_anchor'][i_sample]
    start_index = sum(batch['num_stroke_per_anchor'][:i_sample])
    anc_sample = batch['stroke_wise_split_anchor'][start_index: start_index + anc_stroke_num].numpy()

    pos_vec = batch['sketch_positive_vector']
    pos_img_orig = Image.fromarray(255 - rasterize_Sketch(pos_vec[i_sample].numpy())).convert('RGB')
    pos_stroke_num = batch['num_stroke_per_positive'][i_sample]
    start_index = sum(batch['num_stroke_per_positive'][:i_sample])
    pos_sample = batch['stroke_wise_split_positive'][start_index: start_index + pos_stroke_num].numpy()

    im = Image.new('RGB', (266 * anc_stroke_num, 5 + 261 * 2))  # width x height
    for i_stroke in range(anc_stroke_num):
        anc_img = anc_img_orig.copy()
        draw_anc = ImageDraw.Draw(anc_img)
        anc_stroke_points = np.where(255 - rasterize_Sketch(anc_sample[i_stroke]) == 0)
        for i_point in range(len(anc_stroke_points[0])):
            draw_anc.point((anc_stroke_points[1][i_point],anc_stroke_points[0][i_point]), fill='red')

        pos_img = pos_img_orig.copy()
        draw_pos = ImageDraw.Draw(pos_img)
        pos_stroke_points = np.where(255 - rasterize_Sketch(pos_sample[anc_max[i_stroke].item()]) == 0)
        for i_point in range(len(pos_stroke_points[0])):
            draw_pos.point((pos_stroke_points[1][i_point], pos_stroke_points[0][i_point]), fill='blue')

        im.paste(anc_img, (5+i_stroke*266, 5))      #width x height
        im.paste(pos_img, (5+i_stroke*266, 266))

    im.save(f'{save_path}/{time_id}/Step_{step}.png')

    return
