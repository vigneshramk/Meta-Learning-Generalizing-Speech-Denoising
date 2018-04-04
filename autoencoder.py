import sys
import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
#from torch.distributions import Categorical
from LoadNoise import LoadData
from torch.utils.data import DataLoader

def np_to_variable(x, requires_grad=False, dtype=torch.FloatTensor):
    v = Variable(torch.from_numpy(x).type(dtype), requires_grad=requires_grad)
    if torch.cuda.is_available():
        v = v.cuda()
    return v


class Autoencoder(nn.Module):
    def __init__(self, input_size,hidden_size):
        super(Autoencoder, self).__init__()
        self.hidden_size = hidden_size

        self.W1 = nn.Parameter(torch.rand(input_size, self.hidden_size))

    def forward(self, x):
        self.encoder = F.linear(x, self.W1)
        self.decoder = F.linear(x, self.W1.t())
        x = F.sigmoid(self.encoder)
        x = self.decoder

        return x

class Denoise():

    def __init__(self,model,lr):
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    def train_normal(self,data_full_noisy,data_full_clean):

        num_data = data_full_noisy.shape[0]

        for i in range(num_data):

            noisy = data_full_noisy[i]
            clean = data_full_clean[i]

            noisy = np_to_variable(noisy, requires_grad=True)
            clean = np_to_variable(clean, requires_grad=False)

            self.loss = self.criterion(noisy, clean)
            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()

        return self.loss.data[0]


    def train_maml(self):

        pass


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-epochs', dest='num_epochs', type=int,
                        default=1000, help="Number of epochs to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=1e-4, help="The actor's learning rate.")
    parser.add_argument('--batch-size', type=int,
                        default=64, help="Batch size")
    parser.add_argument('--hidden_size', type=int,
                        default=200, help="hidden size")
    parser.add_argument('--clean_dir', type=str, default='/Users/tylervgina/DataSets/LDC93S1/TIMITcorpus/TIMIT/TRAIN/', metavar='N',
                    help='Clean training files')
    parser.add_argument('--meta_training_file', type=str, default='dataset/meta_data/train/train.txt', metavar='N',
                    help='meta training text file')
    parser.add_argument('--reg_training_file', type=str, default='dataset/reg_data/train/train.txt', metavar='N',
                    help='training text file')
    parser.add_argument('--model', type=int, default= 0, metavar = 'N',
                    help='Which model to use - assuming we are testing different architectures')
    parser.add_argument('--exp_name' ,type=str, default= 'test', metavar = 'N',
                    help='Name of the experiment/weights saved ')                
    parser.add_argument('--frame_size' ,type=int, default = 11, metavar = 'N',
                    help='How many slices we want ')
    parser.add_argument('--SNR', type=int, default=-10, metavar='N',
                    help='how much SNR to add to test')
    parser.add_argument('--noise_type', type=str, default='babble', metavar='N',
                    help='type of noise to add to test')
    parser.add_argument('--clean_dir_test', type=str, default='/Users/tylervgina/DataSets/LDC93S1/TIMITcorpus/TIMIT/TEST/', metavar='N',
                    help='Clean testing files')
    parser.add_argument('--meta_testing_file', type=str, default='dataset/meta_data/test/train.txt', metavar='N',
                    help='meta testing text file')
    parser.add_argument('--reg_testing_file', type=str, default='dataset/reg_data/test/train.txt', metavar='N',
                    help='testing text file')

    # # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    # parser_group = parser.add_mutually_exclusive_group(required=False)
    # parser_group.add_argument('--render', dest='render',
    #                         action='store_true',
    #                         help="Whether to render the environment.")
    # parser_group.add_argument('--no-render', dest='render',
    #                         action='store_false',
    #                         help="Whether to render the environment.")
    # parser.set_defaults(render=False)

    return parser.parse_args()


def main(args):

    args = parse_arguments()
    num_epochs = args.num_epochs
    lr = args.lr
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    clean_dir = args.clean_dir
    meta_training_file = args.meta_training_file
    reg_training_file = args.reg_training_file
    exp_name = args.exp_name
    frame_size = args.frame_size
    noise_type = args.noise_type
    SNR = args.SNR
    reg_clean_test = args.clean_dir_test
    meta_test_file = args.meta_testing_file
    reg_test_file = args.reg_testing_file

    num_samples = 1000
    num_features = 200

    ae_model = Autoencoder(num_features, hidden_size)
    if torch.cuda.is_available():
        ae_model.cuda()

    ae_model.train()

    # Do the data loading here with the given matrix sizes
    data_full_noisy = np.random.rand(num_samples,num_features)
    data_full_clean = np.random.rand(num_samples,num_features)

    #one data loader for each SNR
    meta_training_data_1 = LoadData(tsv_file=meta_training_file, clean_dir=clean_dir,frame_size = frame_size,SNR=-6,noise=noise_type)
    meta_training_data_2 = LoadData(tsv_file=meta_training_file, clean_dir=clean_dir,frame_size = frame_size,SNR=-3,noise=noise_type)
    meta_training_data_3 = LoadData(tsv_file=meta_training_file, clean_dir=clean_dir,frame_size = frame_size,SNR=0,noise=noise_type)
    meta_training_data_4 = LoadData(tsv_file=meta_training_file, clean_dir=clean_dir,frame_size = frame_size,SNR=3,noise=noise_type)
    meta_training_data_5 = LoadData(tsv_file=meta_training_file, clean_dir=clean_dir,frame_size = frame_size,SNR=6,noise=noise_type)

    reg_training_data = LoadData(tsv_file=reg_training_file,clean_dir=clean_dir,frame_size = frame_size,noise=noise_type)

    #ACTUAL DATA LOADERS for each meta/reg

    meta1_train_loader_1 = DataLoader(meta_training_data_1,batch_size=batch_size,shuffle=True,num_workers=0)
    meta_train_loader_2 = DataLoader(meta_training_data_2,batch_size=batch_size,shuffle=True,num_workers=0)
    meta_train_loader_3 = DataLoader(meta_training_data_3,batch_size=batch_size,shuffle=True,num_workers=0)
    meta_train_loader_4 = DataLoader(meta_training_data_4,batch_size=batch_size,shuffle=True,num_workers=0)
    meta_train_loader_5 = DataLoader(meta_training_data_5,batch_size=batch_size,shuffle=True,num_workers=0)

    reg_train_loader = DataLoader(reg_training_data,batch_size=batch_size,shuffle=True,num_workers=0)
    
    dae = Denoise(ae_model,lr)

    for i in range(num_epochs):

        loss = dae.train_normal(data_full_noisy,data_full_clean)
        
        print('epoch [{}/{}], MSE_loss:{:.4f}'
          .format(i + 1, num_epochs, loss))
        break




if __name__ == '__main__':
    main(sys.argv)

