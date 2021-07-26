import argparse
import torch
from model import Sketch_Classification
from dataset import get_dataloader
from stroke_visualiser import show
# from pinakinathc_py import SendEmail

# client = SendEmail()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    expt_name = 'Sketch Semantic Meaning'
    parser = argparse.ArgumentParser(description=expt_name)

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
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--nThreads', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--eval_freq_iter', type=int, default=500)   # 500
    parser.add_argument('--print_freq_iter', type=int, default=5)
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

        # with torch.no_grad():
        #     best_accuracy = model.evaluate(dataloader_Test)

        for epoch in range(hp.max_epoch):
            for i_batch, batch in enumerate(dataloader_Train):
                loss_ncn = model.train_model(batch)
                step += 1

                if (step + 0) % hp.print_freq_iter == 0:
                    print(f'Epoch: {epoch:0>3} | Iter: {i_batch:0>5} | Steps: {step:0>5} | '
                          f'Best_accuracy: {best_accuracy:.5f} | Loss_NCN: {loss_ncn:.5f}')

                if (step + 1) % hp.eval_freq_iter == 0:
                    show(step, batch, model)
                    with torch.no_grad():
                        accuracy = model.evaluate(dataloader_Test)
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        torch.save(model.state_dict(), 'model_best_' + str(hp.training) + '.pth')

        print ('Finished Training')

    # except Exception as e:
    #     message = '\n'.join([expt_name, 'Exception : ', str(e)])
    #     print(message)
    #     if hp.disable_tqdm:
    #         client.send('saneeshan95@gmail.com', message)
    #         client.send('2ajay.das@gmail.com', message)
