from datetime import datetime
import os
import numpy as np
from torchvision.utils import save_image, make_grid
import argparse
import torch
import time
from utils import send_email
from model import FGSBIR_Model
from dataset_QD import get_data, get_dataloader, get_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Category Level SBIR")
    parser.add_argument("--base_dir", type=str, default=os.getcwd(), help="In order to access from condor")
    parser.add_argument("--data_dir", type=str, default='./../Dataset/', help="In order to access from condor")
    parser.add_argument("--saved_models", type=str, default="./models", help="Saved models directory")
    parser.add_argument("--dataset_name", type=str, default="QuickDraw_25", choices=["TUBerlin", "Sketchy", "QuickDraw_25"])
    parser.add_argument("--backbone_name", type=str, default="VGG", help="VGG / InceptionV3/ Resnet50")
    parser.add_argument("--batchsize", type=int, default=16)
    parser.add_argument("--nThreads", type=int, default=12)
    parser.add_argument("--learning_rate", type=float, default=0.00001)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--eval_freq_iter", type=int, default=500)
    parser.add_argument("--print_freq_iter", type=int, default=10)
    parser.add_argument("--splitTrain", type=float, default=0.9)
    parser.add_argument("--training", type=str,default="sketch", help="sketch / photo / edge")
    parser.add_argument('--aux_lambda', type=float, default=0.01)
    parser.add_argument('--draw_frequency', type=int, default=1000)
    parser.add_argument('--disable_tqdm', action='store_true')

    hp = parser.parse_args()
    print(hp, '\n')

    if hp.disable_tqdm:     # Lowering CPU pressure if running on condor
        hp.nThreads = 8

    database = get_data(hp)
    dataloader_Train = get_dataloader(database, mode='Train')
    dataloader_Test = get_dataloader(database, mode='Test')
    dataloader_Test_photo = get_dataloader(database, mode='Test_photo')

    model = FGSBIR_Model(hp)
    model.to(device)
    loadname = f'model_best_{hp.training}.pth'

    if os.path.exists(hp.saved_models):
        # pass
        if os.path.exists(hp.saved_models + '/' + loadname) and hp.disable_tqdm:
            model.load_state_dict(torch.load(
                hp.saved_models + '/' + loadname, map_location=device))
            print(f'\nModel loaded from: {loadname}\n')
    else:
        os.makedirs(hp.saved_models)

    step, best_map = 0, 0.0
    time_id = datetime.now().strftime("%b-%d_%H:%M:%S")

    with torch.no_grad():
        valid_data = model.evaluate(dataloader_Test, dataloader_Test_photo)  
        best_map = np.mean(valid_data["aps@all"])

    print('\nTraining begins:\n')
    for epoch in range(hp.max_epoch):

        for i_batch, batch in enumerate(dataloader_Train):
            # pdb.set_trace()
            # triplet_pair = torch.cat((batch["sketch_img"], batch["positive_img"], batch["negative_img"]), dim=0)
            # save_image(triplet_pair, "triplet_pair_QD.png")
            start = time.time()
            loss = model.train_model(batch)
            step += 1

            if step % hp.print_freq_iter == 0:
                print(f'Epoch: {epoch:0>3} | Iteration: {step:0>5} | '
                      f'Triplet_Loss: {loss:.5f} | '
                      f'mAP_100: {best_map:.5f} | Time: {(time.time() - start):.5f}')

            if step % hp.eval_freq_iter == 0:
                with torch.no_grad():
                    valid_data = model.evaluate(dataloader_Test, dataloader_Test_photo) 
                    map_ = np.mean(valid_data["aps@all"])

                print("mAP@all on validation set after {0} epochs: {1:.4f} (real)".format(epoch + 1, map_))

                if map_ > best_map:
                    best_map = map_
                    torch.save(model.state_dict(), os.path.join(hp.saved_models, "model_best_" + str(hp.training) + ".pth"),)
