import argparse
import os
import torch
import math
from model import Sketch_Classification
from dataset import get_dataloader
from stroke_visualiser import show
from utils import send_email

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    expt_name = 'Sketch Semantic Meaning'
    parser = argparse.ArgumentParser(description=expt_name)

    parser.add_argument('--base_dir', type=str, default='.')
    parser.add_argument('--saved_models', type=str, default='models')
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--bidirectional', type=bool, default=True)
    parser.add_argument('--sketch_LSTM_num_layers', type=int, default=1)
    parser.add_argument('--stroke_LSTM_num_layers', type=int, default=1)
    parser.add_argument('--dropout_stroke', type=float, default=0.)
    parser.add_argument('--dropout_sketch', type=float, default=0.)
    parser.add_argument('--dataset_name', type=str,  default='Sketchy', help='TUBerlin vs Sketchy')
    parser.add_argument('--pool', type=bool, default=False, help='Use Max pooling in Neighbourhood Consensus')
    parser.add_argument('--k_size', type=bool, default=4, help='Kernel Size for Neighbourhood Consensus Network')
    parser.add_argument('--data_encoding_type', type=str, default='3point', help='3point vs 5point')
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--nThreads', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--eval_freq_iter', type=int, default=500)   # 500
    parser.add_argument('--print_freq_iter', type=int, default=10)
    parser.add_argument('--splitTrain', type=int, default=0.7)
    parser.add_argument('--use_conv', type=bool, default=False, help="Whether to use Conv consensus")
    parser.add_argument('--training', type=str, default='sketch', help='sketch / rgb / edge')
    parser.add_argument('--disable_tqdm', action='store_true')

    if True:
        hp = parser.parse_args()
        dataloader_Train, dataloader_Test = get_dataloader(hp)
        print(hp)

        model = Sketch_Classification(hp)
        model.to(device)
        step, best_accuracy = 0, 0

        loadname = f'model_best_{hp.training}.pth'
        if os.path.exists(hp.saved_models):
            if os.path.exists(hp.saved_models + '/' + loadname) and hp.disable_tqdm:
                model.load_state_dict(torch.load(hp.saved_models + '/' + loadname, map_location=device))
                print(f'\nModel loaded from: {loadname}\n')
        else:
            os.makedirs(hp.saved_models)

        # with torch.no_grad():
        #     best_accuracy = model.evaluate(dataloader_Test)

        pos, neg = 0, 0
        for epoch in range(hp.max_epoch):
            for i_batch, batch in enumerate(dataloader_Train):
                loss_ncn = model.train_model(batch)
                step += 1
                pos += int(loss_ncn < 0)     # loss = neg - pos; hence reversed
                neg += int(loss_ncn > 0)

                if (step + 0) % hp.print_freq_iter == 0:
                    print(f'Epoch: {epoch:0>3} | Iter: {i_batch:0>5} | Steps: {step:0>5} | Best_accuracy: {best_accuracy:.5f} | pos:neg = {int(pos/math.gcd(pos,neg)):0>3}:{int(neg/math.gcd(pos,neg)):0>3} | Loss_NCN: {loss_ncn:.5f}')

                if (step + 1) % hp.eval_freq_iter == 0:
                    with torch.no_grad():
                        accuracy = model.evaluate(dataloader_Test)
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        show(step, batch, model, hp.saved_models)
                        torch.save(model.state_dict(), os.path.join(hp.saved_models, "model_best_" + str(hp.training) + ".pth"))

        print ('Finished Training')

    # except Exception as e:
    #     message = '\n'.join([expt_name, 'Exception : ', str(e)])
    #     print(message)
    #     if hp.disable_tqdm:
    #         send_email('saneeshan95@gmail.com', message)
