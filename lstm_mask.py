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
from TestAddNoiseLoader import TestSpect
import utils
import matplotlib.pyplot as plt

from adam_new import Adam_Custom
from copy import deepcopy
import time

use_cuda = torch.cuda.is_available()

print('Cuda')
print(use_cuda)
#CUDA_VISIBLE_DEVICES=2,3
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

def test_mask(model,clean,noise):
    criterion = nn.MSELoss()
    noise_batch = np_to_variable(noise)
    clean_batch = np_to_variable(clean)

    approx_clean = model(noise_batch)
    loss = criterion(approx_clean, clean_batch)
    mse = loss.data[0]
    return approx_clean.data.cpu().numpy().T, mse

def np_to_variable(x, requires_grad=False, dtype=torch.FloatTensor):
    v = Variable(torch.from_numpy(x).type(dtype), requires_grad=requires_grad)
    if torch.cuda.is_available():
        v = v.cuda()
    return v

class LSTM_Mask(nn.Module):
    #make argparse dropout
    def __init__(self, input_size = 161, hidden_size = 256 ,num_layers = 2, dropout = 0 , bidirectional = False):
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

        #Add L2 regularization through weight decay
        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=train_lr,weight_decay=0.5)
        #make this arg parse
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=train_lr,weight_decay=1e-6)
        # self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=meta_lr)

        self.meta_optimizer = Adam_Custom(self.model.parameters(), lr=meta_lr,weight_decay=1e-6)

        self.stamp = time.strftime("%Y%m%d-%H%M%S")

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

        # if j%5==0 and i==0:
        # now saving model for every epoch with time stamp
        state = {
            'epoch': j,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

        str_path = model_path + '/model_lstm_'+ str(self.stamp) + '_' + str(j) + '.h5'
        torch.save(state,str_path)
        # print("Saving the model")

        return self.loss.data[0]


    def train_maml(self,meta_train_noisy,meta_train_clean,train_datapts,meta_train_datapts,num_iter,test_file,file_name,noise_type):

        num_tasks,num_data,_,_ = meta_train_noisy.shape

        path_name = './figures/maml_train_plots/' + '/'
        str_path1 = 'training_loss_maml_mask_lstm_total_' + file_name + '.png'
        plot1_name = os.path.join(path_name,str_path1)

        model_path = 'models/lstm_mask_maml_train/' + file_name 

        print(model_path)
        
        if not os.path.exists(path_name):
            os.makedirs(path_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        loader = TestSpect('dataset/meta_data/test/test.txt',test_file, SNR=-6, noise=noise_type)
        print('Testing '+ noise_type)
        test_loader = DataLoader(loader,batch_size=1,shuffle=True,num_workers=0)

        test_error_all = []

        K = train_datapts
        D = meta_train_datapts

        theta_list = []

        num_epoch = 0
        for i in range(num_iter):

            if(i%1000 == 0):
                num_epoch +=1

            # Get the theta
            if i == 0:
                theta= self.get_weights()

            # Individual gradient updates theta_i's ---Training mode
            for t in range(num_tasks):

                #Sample K datapoints from the task t
                idx_train = np.random.randint(num_data,size=K)

                noisy = meta_train_noisy[t,idx_train,:,:]
                clean = meta_train_clean[t,idx_train,:,:]

                noisy = np_to_variable(noisy, requires_grad=True)
                clean = np_to_variable(clean, requires_grad=False)

                output1 = self.model(noisy)

                # Initialize the network with current network weights
                self.set_weights(theta)

                # Train the network with the given K samples
                self.loss = self.criterion(output1, clean)
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
            theta_copy = deepcopy(theta)  
            for t in range(num_tasks):

                #Sample K datapoints from the task t
                idx_meta = np.random.randint(num_data,size=D)

                noisy = meta_train_noisy[t,idx_meta,:,:]
                clean = meta_train_clean[t,idx_meta,:,:]

                noisy = np_to_variable(noisy, requires_grad=True)
                clean = np_to_variable(clean, requires_grad=False)

                # output2 = self.model(noisy)

                #Get the loss w.r.t the theta_i network
                self.set_weights(theta_list[t])
                approx_clean = self.model(noisy)
                self.loss_outer = self.criterion(approx_clean, clean)

                # Set the model weights to theta before training
                #Train with this theta on the D samples
                self.set_weights(theta_copy)
                self.meta_optimizer.zero_grad()
                grads = torch.autograd.grad(self.loss_outer, self.model.parameters())
                #Pass the gradients directly to the Custom Adam optimizer
                self.set_weights(theta)
                self.meta_optimizer.step(grads)
            
                
                # self.set_weights(theta)
                # self.meta_optimizer.zero_grad()
                # self.loss.backward()
                # self.meta_optimizer.step()

                # Theta will now have the updated parameters
                theta = self.get_weights()

                #Add up the losses from each of these networks
                combined_loss += self.loss_outer.data[0]

            print("Average Loss in iteration %s is %.4f" %(i,combined_loss/num_tasks))

            if (i%1000 == 0):
                print('Epoch %s done' %num_epoch)
                state = {
                    'epoch': num_epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.meta_optimizer.state_dict(),
                }
                str_path = model_path + '/maml_lstm_' + str(num_epoch) + '.h5'
                torch.save(state,str_path)
                print("Saving the model")

                print('Testing....')
                test_error = []
                for j, batch in enumerate(test_loader):
                    if(j==0):
                       break
                    #print('Testing File: %d' % i)
                    #get the clean magnitudes and the noise magnitude at the specific SNR
                    clean_mag = batch['clean_mag'].numpy()
                    noise_mag = batch['noise_mag'].numpy()
                    
                    _ , mse = test_mask(self.model, clean_mag, noise_mag)
                    test_error.append(mse)

                test_error_all.append(np.mean(test_error))
                print(test_error_all)


def main(args):

    args = utils.parse_arguments()
    num_epochs = args.num_epochs
    train_lr = args.train_lr
    meta_lr = args.meta_lr
    exp_name = args.exp_name
    noise_type = args.noise_type
    train_all = args.train_all
    reg_train = args.reg_train
    num_spect = args.num_spectograms
    save_name = args.save_file_name
    test_file = args.test_file

    train_datapts = 99
    meta_train_datapts = 99

    num_iter = 100000

    model = LSTM_Mask()
    if torch.cuda.is_available():
        model.cuda()

    model.train()
    dae = Denoise(model,train_lr,meta_lr)

    if reg_train == 1:
        print('Regular Training.....')
        # Create plot
        fig1 = plt.figure()
        ax1 = fig1.gca()
        ax1.set_title('Loss vs Epochs')

        if train_all == 1:
            print('Training All....')
            #all_noise = ['engine','factory1','babble']
            all_noise =['factory1','babble']#,'engine','ops'] # ONLY CHANGE THIS ONE. change to whatever noise types you want to train with 
            file_name = exp_name  
            print(all_noise)
        else:
            print('Training ' + noise_type)
            all_noise = [noise_type]
            file_name = exp_name

        noisy_total = []
        clean_total =[]

        for n in all_noise:
            print(n)
            noisy_data1 = np.load('spectograms/spectograms_train'+str(num_spect)+'/noise/' + n + '/train/noise_-6.npy')
            noisy_data2 = np.load('spectograms/spectograms_train'+str(num_spect)+'/noise/'+ n + '/train/noise_-3.npy')
            noisy_data3 = np.load('spectograms/spectograms_train'+str(num_spect)+'/noise/' + n + '/train/noise_0.npy')
            noisy_data4 = np.load('spectograms/spectograms_train'+str(num_spect)+'/noise/'+ n + '/train/noise_3.npy')
            noisy_data5 = np.load('spectograms/spectograms_train'+str(num_spect)+'/noise/' + n + '/train/noise_6.npy')

            clean_data = np.load('spectograms/spectograms_train'+str(num_spect)+'/clean/train/clean_frames_' + n + '.npy')
        
            noisy_sq1 = np.reshape(noisy_data1,[noisy_data1.shape[0]*noisy_data1.shape[1],noisy_data1.shape[2],noisy_data1.shape[3]])
            noisy_sq2 = np.reshape(noisy_data2,[noisy_data2.shape[0]*noisy_data2.shape[1],noisy_data2.shape[2],noisy_data2.shape[3]])
            noisy_sq3 = np.reshape(noisy_data3,[noisy_data3.shape[0]*noisy_data3.shape[1],noisy_data3.shape[2],noisy_data3.shape[3]])
            noisy_sq4 = np.reshape(noisy_data4,[noisy_data4.shape[0]*noisy_data4.shape[1],noisy_data4.shape[2],noisy_data4.shape[3]])
            noisy_sq5 = np.reshape(noisy_data5,[noisy_data5.shape[0]*noisy_data5.shape[1],noisy_data5.shape[2],noisy_data5.shape[3]])
        
            noisy_total.extend(noisy_sq1)
            noisy_total.extend(noisy_sq2)
            noisy_total.extend(noisy_sq3)
            noisy_total.extend(noisy_sq4)
            noisy_total.extend(noisy_sq5)

            clean_sq1 = np.reshape(clean_data,[clean_data.shape[0]*clean_data.shape[1],clean_data.shape[2],clean_data.shape[3]])
            clean_total.extend(clean_sq1)
            clean_total.extend(clean_sq1)
            clean_total.extend(clean_sq1)
            clean_total.extend(clean_sq1)
            clean_total.extend(clean_sq1)

        noisy_total = np.array(noisy_total)
        print(noisy_total.shape)

        clean_total = np.array(clean_total)
        print(clean_total.shape)
        path_name = './figures/train_plots/' + file_name + '/'
        str_path1 = 'training_loss_normal_mask_lstm_total_' + exp_name + '.png'
        plot1_name = os.path.join(path_name,str_path1)

        model_path = 'models/lstm_mask_normal_train/' + file_name 

        print(model_path)
        
        if not os.path.exists(path_name):
            os.makedirs(path_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # Normal training with one SNR
        num_samples = int(noisy_total.shape[0])
        num_batches = num_samples/128
        loader = TestSpect('dataset/meta_data/test/test.txt',test_file, SNR=-3, noise=noise_type)
        print('Testing '+ noise_type)
        test_loader = DataLoader(loader,batch_size=1,shuffle=True,num_workers=0)

        test_error_all = []
        print('Training.....')
        for j in range(num_epochs):

            # test after every epochs
            print('Testing....')
            test_error = []
            for i, batch in enumerate(test_loader):
                # only test first 50 sample. This is randrom for every epochs
                if(i==50):
                    break
                #get the clean magnitudes and the noise magnitude at the specific SNR
                clean_mag = batch['clean_mag'].numpy()
                noise_mag = batch['noise_mag'].numpy()
                
                _ , mse = test_mask(dae.model, clean_mag, noise_mag)
                test_error.append(mse)

            test_error_all.append(np.mean(test_error))
            print(test_error_all)

            shuffle_idx = np.random.permutation(noisy_total.shape[0])

            noisy_total = noisy_total[shuffle_idx]
            clean_total = clean_total[shuffle_idx]

            total_loss = 0
            step = 128
            for i in range(0,num_samples-step,step):
                clean = clean_total[i:i+step,:]
                noise = noisy_total[i:i+step,:]
                # noise = np.log(noise)
                if(noise.shape[0] is not 0):
                    loss = dae.train_normal(noise,clean,j+1,i,model_path)

                # print("Batch - %s : %s , Loss - %1.4f" %(i, i+step,loss))

                total_loss += loss
                
            print('epoch [{}/{}], MSE_loss:{:.8f}'.format(j + 1, num_epochs, total_loss/num_batches))
            ax1.scatter(j+1, total_loss)
            if j%100 == 0:
                ax1.figure.savefig(plot1_name)
        print(test_error_all)
    else:
        print("meta training.....")

        all_noise = ['babble','factory1','engine','bucc','ops','bike']
        all_babble_noise = []
        all_babble_clean = []
        all_factory1_noise = []
        all_factory1_clean = []
        all_engine_noise = []
        all_engine_clean = []
        all_bucc_noise = []
        all_bucc_clean = []
        all_ops_noise = []
        all_ops_clean = []
        all_bike_noise = []
        all_bike_clean = []
        
        for n in all_noise:
            print(n)
            noisy_total = []
            clean_total = []

            noisy_data1 = np.load('spectograms/spectograms_train'+str(num_spect)+'/noise/' + n + '/train/noise_-6.npy')
            noisy_data2 = np.load('spectograms/spectograms_train'+str(num_spect)+'/noise/'+ n + '/train/noise_-3.npy')
            noisy_data3 = np.load('spectograms/spectograms_train'+str(num_spect)+'/noise/' + n + '/train/noise_0.npy')
            noisy_data4 = np.load('spectograms/spectograms_train'+str(num_spect)+'/noise/'+ n + '/train/noise_3.npy')
            noisy_data5 = np.load('spectograms/spectograms_train'+str(num_spect)+'/noise/' + n + '/train/noise_6.npy')

            clean_data = np.load('spectograms/spectograms_train'+str(num_spect)+'/clean/train/clean_frames_' + n + '.npy')
        
            noisy_sq1 = np.reshape(noisy_data1,[noisy_data1.shape[0]*noisy_data1.shape[1],noisy_data1.shape[2],noisy_data1.shape[3]])
            noisy_sq2 = np.reshape(noisy_data2,[noisy_data2.shape[0]*noisy_data2.shape[1],noisy_data2.shape[2],noisy_data2.shape[3]])
            noisy_sq3 = np.reshape(noisy_data3,[noisy_data3.shape[0]*noisy_data3.shape[1],noisy_data3.shape[2],noisy_data3.shape[3]])
            noisy_sq4 = np.reshape(noisy_data4,[noisy_data4.shape[0]*noisy_data4.shape[1],noisy_data4.shape[2],noisy_data4.shape[3]])
            noisy_sq5 = np.reshape(noisy_data5,[noisy_data5.shape[0]*noisy_data5.shape[1],noisy_data5.shape[2],noisy_data5.shape[3]])
        
            noisy_total.append(noisy_sq1)
            noisy_total.append(noisy_sq2)
            noisy_total.append(noisy_sq3)
            noisy_total.append(noisy_sq4)
            noisy_total.append(noisy_sq5)
        
            clean_sq1 = np.reshape(clean_data,[clean_data.shape[0]*clean_data.shape[1],clean_data.shape[2],clean_data.shape[3]])

            clean_total.append(clean_sq1)
            clean_total.append(clean_sq1)
            clean_total.append(clean_sq1)
            clean_total.append(clean_sq1)
            clean_total.append(clean_sq1)

            if n == 'factory1':
                print('factory1 copy')
                all_factory1_noise = noisy_total
                all_factory1_clean = clean_total

            elif n == 'babble':
                print('babble copy ')
                all_babble_noise = noisy_total
                all_babble_clean =  clean_total

            elif n == 'engine':
                print('engine copy ')
                all_engine_noise =  noisy_total
                all_engine_clean =  clean_total
            elif n == 'bucc':
                print('bucc copy ')
                all_bucc_noise =  noisy_total
                all_bucc_clean =  clean_total
            elif n == 'ops':
                print('ops copy ')
                all_ops_noise =  noisy_total
                all_ops_clean =  clean_total
            elif n == 'bike':
                print('bike copy ')
                all_bike_noise =  noisy_total
                all_bike_clean =  clean_total

        print("Babble shape")
        print(all_babble_noise.shape,all_babble_clean.shape)
        print("Factory shape")
        print(all_factory1_noise.shape,all_factory1_clean.shape)
        print("Engine shape")
        print(all_engine_noise.shape,all_engine_clean.shape)
        print("Bucc shape")
        print(all_bucc_noise.shape,all_bucc_clean.shape)
        print("ops shape")
        print(all_ops_noise.shape,all_ops_clean.shape)
        print("bike shape")
        print(all_bike_noise.shape,all_bike_clean.shape)

        maml_noisy_data = []
        maml_clean_data = []

        maml_noisy_data.extend(all_babble_noise)
        maml_noisy_data.extend(all_factory1_noise)
        maml_noisy_data.extend(all_engine_noise)
        maml_noisy_data.extend(all_bucc_noise)
        maml_noisy_data.extend(all_ops_noise)
        maml_noisy_data.extend(all_bike_noise)

        maml_clean_data.extend(all_babble_clean)
        maml_clean_data.extend(all_factory1_clean)
        maml_clean_data.extend(all_engine_clean)
        maml_clean_data.extend(all_bucc_clean)
        maml_clean_data.extend(all_ops_clean)
        maml_clean_data.extend(all_bike_clean)

        maml_noisy_data = np.array(maml_noisy_data)
        maml_clean_data = np.array(maml_clean_data)

        print("Size of total training")
        print(maml_noisy_data.shape)
        print(maml_clean_data.shape)

        # print(all_factory1_noise.shape)

        # maml_data_noise = np.zeros((len(all_noise),all_babble_noise.shape[0],all_babble_noise.shape[1],all_babble_noise.shape[2]))
        # maml_data_clean = np.zeros((len(all_noise),all_babble_noise.shape[0],all_babble_noise.shape[1],all_babble_noise.shape[2]))

        # maml_data_noise[0] = all_babble_noise
        # maml_data_noise[1] = all_factory1_noise

        # maml_data_clean[0] = all_babble_clean
        # maml_data_clean[1] = all_factory1_clean 

        # print(maml_data_noise.shape)
        # print(maml_data_clean.shape)

        file_name = exp_name

        #Meta-training with five SNR
        dae.train_maml(maml_noisy_data,maml_clean_data,train_datapts,meta_train_datapts,num_iter,test_file,file_name,noise_type)



if __name__ == '__main__':
    main(sys.argv)

