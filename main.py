import torch
import time
from model import Sketch_Classification
from dataset import get_dataloader
# from sketch_dataloader import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import argparse
import numpy as np
# from RGB_dataset import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Sketch Semantic Meaning')

    parser.add_argument('--hidden_size', type=int, default= 512)
    parser.add_argument('--bidirectional', type=bool, default=True)
    parser.add_argument('--sketch_LSTM_num_layers', type=int, default=1)
    parser.add_argument('--stroke_LSTM_num_layers', type=int, default=1)
    parser.add_argument('--dropout_stroke', type=float, default=0.)
    parser.add_argument('--dropout_sketch', type=float, default=0.)


    parser.add_argument('--dataset_name', type=str, default='Sketchy', help = 'TUBerlin vs Sketchy')
    parser.add_argument('--data_encoding_type', type=str, default='3point', help='3point vs 5point')
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--nThreads', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--eval_freq_iter', type=int, default=1000)
    parser.add_argument('--print_freq_iter', type=int, default=10)
    parser.add_argument('--splitTrain', type=int, default=0.7)
    parser.add_argument('--training', type=str, default='sketch', help = 'sketch / rgb / edge')

    hp = parser.parse_args()
    dataloader_Train, dataloader_Test = get_dataloader(hp)

    # print(hp)
    # dataloader_Train_Sketch, dataloader_Test_Sketch = get_dataloader_sketch(hp)


    model = Sketch_Classification(hp)
    model.to(device)
    step=  0
    best_accuracy = 0

    for epoch in range(hp.max_epoch):

        for i_batch, batch in enumerate(dataloader_Train):
            loss = model.train_model(batch)
            step += 1

            if (step + 0) % hp.print_freq_iter == 0:
                print('Epoch: {}, Iter: {}, Steps: {}, Loss: {}, Best Accuracy: {}'.format(epoch, i_batch, step, loss,
                                                                                           best_accuracy))

            if (step + 1) % hp.eval_freq_iter == 0:
                with torch.no_grad():
                    accuracy = model.evaluate(dataloader_Test)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    torch.save(model.state_dict(), 'model_best_' + str(hp.training) + '.pth')

