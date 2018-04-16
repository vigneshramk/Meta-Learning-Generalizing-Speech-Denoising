import matplotlib
matplotlib.use('Agg')
import sys
import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from LoadNoise import LoadData
from torch.utils.data import DataLoader
import utils

import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()

print('Cuda')
print(use_cuda)

FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

def np_to_variable(x, requires_grad=False, dtype=torch.FloatTensor):
    v = Variable(torch.from_numpy(x).type(dtype), requires_grad=requires_grad)
    if torch.cuda.is_available():
        v = v.cuda()
    return v

class LSTM_Mask(nn.Module):
    def __init__(self, input_size = 161, hidden_size = 256 ,num_layers = 2,dropout = False, bidirectional = False):
        super(LSTM_Mask, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,batch_first=True,dropout=dropout,bidirectional=bidirectional)
        self.fc1 = nn.Linear(self.hidden_size, input_size)

    def forward(self, x):
        
        out, _ = self.lstm(x)
        output_mask = F.sigmoid(self.fc1(out))
        approx_clean = x * output_mask
        return approx_clean

class Denoise():

    def __init__(self,model,train_lr,meta_lr):

        self.model = model

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=train_lr)
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=meta_lr)

    def get_weights(self):

        curr_model = {'state_dict': self.model.state_dict()}
        
        # if(mode is 'train'):
        #   curr_model = {'state_dict': self.model.state_dict()}
        #                  # 'optimizer': self.optimizer.state_dict()}
        # elif(mode is 'meta'):
        #   curr_model = {'state_dict': self.model.state_dict()}
        #                  # 'optimizer': self.meta_optimizer.state_dict()}
        return curr_model
    
    def set_weights(self,curr_model):

        self.model.load_state_dict(curr_model['state_dict'])

        # if(mode is 'train'):
        #   self.model.load_state_dict(curr_model['state_dict'])
        #   # self.optimizer.load_state_dict(curr_model['optimizer'])
        # elif(mode is 'meta'):
        #   self.model.load_state_dict(curr_model['state_dict'])
        #   # self.meta_optimizer.load_state_dict(curr_model['optimizer'])
        
    def train_normal(self,noisy,clean,j,i,model_path):

        #print(noisy.shape)
        #print(clean.shape)
        noisy_th = np_to_variable(noisy, requires_grad=True)
        clean_th = np_to_variable(clean)

        output_th = self.model(noisy_th)

        self.loss = self.criterion(output_th, clean_th)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        if j%50==0 and i==0:

            state = {
                'epoch': j,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }

            str_path = model_path + '/model_lstm' + '.h5'
            torch.save(state,str_path)
            print("Saving the model")

        return self.loss.data[0]


    def train_maml(self,meta_train_noisy,meta_train_clean,train_datapts,meta_train_datapts,num_iter):

        num_data,num_features,num_tasks = meta_train_noisy.shape

        K = train_datapts
        D = meta_train_datapts

        theta_list = []


        for i in range(num_iter):

            # Get the theta
            if i == 0:
                theta= self.get_weights()

            # Individual gradient updates theta_i's ---Training mode
            for t in range(num_tasks):

                #Sample K datapoints from the task t
                idx_train = np.random.randint(num_data,size=K)

                noisy = meta_train_noisy[idx_train,:,t]
                clean = meta_train_clean[idx_train,:,t]

                noisy = np_to_variable(noisy, requires_grad=True)
                clean = np_to_variable(clean, requires_grad=False)

                # Initialize the network with current network weights
                self.set_weights(theta)

                # Train the network with the given K samples

                approx_clean = self.model(noisy) 
                self.loss = self.criterion(approx_clean, clean)
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()

                #Update params theta_i
                if i == 0:
                    theta_list.append(self.get_weights())
                else:
                    theta_list[t] = self.get_weights()


            # Theta parameter update --- Meta-training mode
            combined_loss = 0   
            for t in range(num_tasks):

                #Sample K datapoints from the task t
                idx_meta = np.random.randint(num_data,size=D)

                noisy = meta_train_noisy[idx_meta,:,t]
                clean = meta_train_clean[idx_meta,:,t]

                noisy = np_to_variable(noisy, requires_grad=True)
                clean = np_to_variable(clean, requires_grad=False)

                #Get the loss w.r.t the theta_i network
                self.set_weights(theta_list[t])
                approx_clean = self.model(noisy) 
                self.loss = self.criterion(approx_clean, clean)

                # Set the model weights to theta before training
                #Train with this theta on the D samples
                self.set_weights(theta)
                self.meta_optimizer.zero_grad()
                self.loss.backward()
                self.meta_optimizer.step()

                theta = self.get_weights()

                #Add up the losses from each of these networks
                combined_loss += self.loss.data[0]

            print("Average Loss in iteration %s is %1.2f" %(i,combined_loss/num_tasks))


def main(args):

    args = utils.parse_arguments()
    num_epochs = args.num_epochs
    train_lr = args.train_lr
    meta_lr = args.meta_lr
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
    num_tasks = 5
    train_datapts = 100
    meta_train_datapts = 100

    num_iter = 10000

    model = LSTM_Mask()
    if torch.cuda.is_available():
        model.cuda()

    model.train()

    # Create plot
    fig1 = plt.figure()
    ax1 = fig1.gca()
    ax1.set_title('Loss vs Epochs')

    noisy_data1 = np.load('spectograms_train30/noise/' + noise_type + '/train/noise_-6.npy')
    noisy_data2 = np.load('spectograms_train30/noise/'+ noise_type + '/train/noise_-3.npy')
    noisy_data3 = np.load('spectograms_train30/noise/' + noise_type + '/train/noise_0.npy')
    noisy_data4 = np.load('spectograms_train30/noise/'+ noise_type + '/train/noise_3.npy')
    noisy_data5 = np.load('spectograms_train30/noise/' + noise_type + '/train/noise_6.npy')

    clean_data = np.load('spectograms_train30/clean/train/clean_frames_' + noise_type + '.npy')
    
    noisy_sq1 = np.reshape(noisy_data1,[noisy_data1.shape[0]*noisy_data1.shape[1],noisy_data1.shape[2],noisy_data1.shape[3]])
    noisy_sq2 = np.reshape(noisy_data2,[noisy_data2.shape[0]*noisy_data2.shape[1],noisy_data2.shape[2],noisy_data2.shape[3]])
    noisy_sq3 = np.reshape(noisy_data3,[noisy_data3.shape[0]*noisy_data3.shape[1],noisy_data3.shape[2],noisy_data3.shape[3]])
    noisy_sq4 = np.reshape(noisy_data4,[noisy_data4.shape[0]*noisy_data4.shape[1],noisy_data4.shape[2],noisy_data4.shape[3]])
    noisy_sq5 = np.reshape(noisy_data5,[noisy_data5.shape[0]*noisy_data5.shape[1],noisy_data5.shape[2],noisy_data5.shape[3]])
    print(noisy_data1.shape)
    print(noisy_sq1.shape)
    noisy_total = []
    
    noisy_total.extend(noisy_sq1)
    noisy_total.extend(noisy_sq2)
    noisy_total.extend(noisy_sq3)
    noisy_total.extend(noisy_sq4)
    noisy_total.extend(noisy_sq5)
    
    noisy_total = np.array(noisy_total)
    print(noisy_total.shape)

    clean_sq1 = np.reshape(clean_data,[clean_data.shape[0]*clean_data.shape[1],clean_data.shape[2],clean_data.shape[3]])

    clean_total =[]

    clean_total.extend(clean_sq1)
    clean_total.extend(clean_sq1)
    clean_total.extend(clean_sq1)
    clean_total.extend(clean_sq1)
    clean_total.extend(clean_sq1)

    clean_total = np.array(clean_total)
    print(clean_total.shape)
   
    dae = Denoise(model,train_lr,meta_lr)

    path_name = './figures/train_plots/' + noise_type + '/'
    str_path1 = 'training_loss_normal_mask_lstm_total_' + exp_name + '.png'
    plot1_name = os.path.join(path_name,str_path1)

    model_path = 'models/lstm_mask_normal_train/' + noise_type

    print(model_path)
    
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Normal training with one SNR
    num_samples = int(noisy_total.shape[0])
    num_batches = num_samples/500

    print('Training.....')
    for j in range(num_epochs):
        shuffle_idx = np.random.permutation(noisy_total.shape[0])

        noisy_total = noisy_total[shuffle_idx]
        clean_total = clean_total[shuffle_idx]

        total_loss = 0
        step = 500
        for i in range(0,num_samples-step,step):
            clean = clean_total[i:i+step,:]
            noise = noisy_total[i:i+step,:]
            # noise = np.log(noise)
            if(noise.shape[0] is not 0):
                loss = dae.train_normal(noise,clean,j+1,i,model_path)

            # print("Batch - %s : %s , Loss - %1.4f" %(i, i+step,loss))

            total_loss += loss
            
        print('epoch [{}/{}], MSE_loss:{:.4f}'.format(j + 1, num_epochs, total_loss/num_batches))
        ax1.scatter(j+1, total_loss)
        if j%100 == 0:
            ax1.figure.savefig(plot1_name)
            
    #Meta-training with five SNR
    # dae.train_maml(meta_train_noisy,meta_train_clean,train_datapts,meta_train_datapts,num_iter)




if __name__ == '__main__':
    main(sys.argv)

