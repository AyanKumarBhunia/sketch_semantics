import torch
import os
import numpy as np
from datetime import datetime
from PIL import Image, ImageDraw
from rasterize import rasterize_Sketch
from model import make_mask

time_id = datetime.now().strftime("%b%d_%H-%M-%S")


def show(step, batch, model):
    if not os.path.isdir(time_id):
        os.makedirs(time_id)

    output_anc, num_stroke_anc = model.Network(batch, type='anchor')  # b,N1,512
    output_pos, num_stroke_pos = model.Network(batch, type='positive') # b,N2,512
    mask_anc, mask_pos = map(make_mask, [num_stroke_anc, num_stroke_pos])
    corr_xpos = model.neighbour(output_anc, output_pos, mask_anc, mask_pos)

    im = Image.new('RGB', (266*output_anc.shape[0], 5+261*2))     # width x height
    '''
    I need the n1 x n2 matrix
    For every element in N1, the top 2 values in N2 for which the N1xN2 matrix value is highest, will be coloured
    '''
    for i_sample, A in enumerate(corr_xpos):
        a, b = torch.where(A == A.max())
        a, b = a[0].item(), b[0].item()       # in the rare case there are multiple same max values

        k = batch['anchor_sketch_vector']
        anchor = Image.fromarray(255-rasterize_Sketch(k[i_sample].numpy())).convert('RGB')
        draw_anc = ImageDraw.Draw(anchor)
        nOFstroke = batch['num_stroke_per_anchor'][i_sample]
        start_index = sum(batch['num_stroke_per_anchor'][:i_sample])
        beta_anc = batch['stroke_wise_split_anchor'][start_index: start_index+nOFstroke].numpy()

        k = batch['sketch_positive_vector']
        positive = Image.fromarray(255 - rasterize_Sketch(k[i_sample].numpy())).convert('RGB')
        draw_pos = ImageDraw.Draw(positive)
        nOFstroke = batch['num_stroke_per_positive'][i_sample]
        start_index = sum(batch['num_stroke_per_positive'][:i_sample])
        beta_pos = batch['stroke_wise_split_positive'][start_index: start_index + nOFstroke].numpy()

        a, b = a % beta_anc.shape[0], b % beta_pos.shape[0]    # not accurate - avoid errors

        beta_stroke_points_anc = np.where(255 - rasterize_Sketch(beta_anc[a]) == 0)
        for i_point in range(len(beta_stroke_points_anc[0])):
            draw_anc.point((beta_stroke_points_anc[1][i_point],beta_stroke_points_anc[0][i_point]), fill='red')

        beta_stroke_points_pos = np.where(255 - rasterize_Sketch(beta_pos[b]) == 0)
        for i_point in range(len(beta_stroke_points_pos[0])):
            draw_pos.point((beta_stroke_points_pos[1][i_point], beta_stroke_points_pos[0][i_point]), fill='blue')

        im.paste(anchor, (5+i_sample*266, 5))      #width x height
        im.paste(positive, (5+i_sample*266, 266))

    im.save(f'{time_id}/Step_{step}.png')

    return
