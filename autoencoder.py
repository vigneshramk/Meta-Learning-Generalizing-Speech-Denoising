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
import matplotlib.pyplot as plt

from adam_new import Adam_Custom

from clip_grad_norm import clip_grad_norm_

def np_to_variable(x, requires_grad=False, dtype=torch.FloatTensor):
	v = Variable(torch.from_numpy(x).type(dtype), requires_grad=requires_grad)
	if torch.cuda.is_available():
		v = v.cuda()
	return v

#(12000*5,161*11) all noisetypes together
#(12000,161*11) one noise type to train/test
#(12000,161*11,5) one noise types to meta train 

#ALL   3 frames each

#we should try just regular fully connected layers like spectral mapping
#paper - https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7067387
#they do it with log magnitude, and they normalize the data

class Auto(nn.Module):
	def __init__(self, input_size, output_size):
		super(Auto, self).__init__()
		self.hidden_size = 1600
		#self.hidden2_size = 750
		#change it to what the paper had. 3 hidden layers 1600
		#but they normlized data and use log magnitude. 
		#got the perfect hidden size and units by cross validation
		self.classifier = nn.Sequential(
						  nn.Linear(input_size, self.hidden_size),
						  nn.ReLU(inplace=True),
						  nn.Linear(self.hidden_size, self.hidden_size),
						  nn.ReLU(inplace=True),
						  nn.Linear(self.hidden_size, self.hidden_size),
						  nn.ReLU(inplace=True),
						  nn.Linear(self.hidden_size, output_size))

	def forward(self, x):
		x = self.classifier(x)
		return x


class Mask(nn.Module):
	def __init__(self, input_size, output_size):
		super(Mask, self).__init__()
		self.hidden_size = 1600
		#self.hidden2_size = 750
		#change it to what the paper had. 3 hidden layers 1600
		#but they normlized data and use log magnitude. 
		#got the perfect hidden size and units by cross validation
		self.classifier = nn.Sequential(
						  nn.Linear(input_size, self.hidden_size),
						  nn.ReLU(inplace=True),
						  nn.Linear(self.hidden_size, self.hidden_size),
						  nn.ReLU(inplace=True),
						  nn.Linear(self.hidden_size, self.hidden_size),
						  nn.ReLU(inplace=True),
						  nn.Linear(self.hidden_size, output_size))

	def forward(self, x):
		x = self.classifier(x)
		return F.sigmoid(x)


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

	def __init__(self,model,train_lr,meta_lr):

		self.model = model

		self.criterion = nn.MSELoss()
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=train_lr)
		self.meta_optimizer = Adam_Custom(self.model.parameters(), lr=meta_lr)


	def get_weights(self):

		curr_model = {'state_dict': self.model.state_dict()}
		
		# if(mode is 'train'):
		#   curr_model = {'state_dict': self.model.state_dict()}
		#                  # 'optimizer': self.optimizer.state_dict()}
		# elif(mode is 'meta'):
		#   curr_model = {'state_dict': self.model.state_dict()}
		#                  # 'optimizer': self.meta_optimizer.state_dict()}
		return curr_model

	# def grad_reverse(grad):
 #        return grad.clone() * -1
	
	def set_weights(self,curr_model):

		self.model.load_state_dict(curr_model['state_dict'])

		# if(mode is 'train'):
		#   self.model.load_state_dict(curr_model['state_dict'])
		#   # self.optimizer.load_state_dict(curr_model['optimizer'])
		# elif(mode is 'meta'):
		#   self.model.load_state_dict(curr_model['state_dict'])
		#   # self.meta_optimizer.load_state_dict(curr_model['optimizer'])
		
	def train_normal(self,noisy,clean,j,i,model_path):

		def grad_reverse(grad):
			return grad.clone()*-1


		noisy_th = np_to_variable(noisy)
		clean_th = np_to_variable(clean)

		mask_th = self.model(noisy_th)

		#mask = mask_th.data.cpu().numpy()

		# noisy_middle = noisy_th[:,161*5:161*6]

		output = mask_th

		# output = np_to_variable(output,requires_grad=True)

		self.loss = self.criterion(output, clean_th)

		grads = torch.autograd.grad(self.loss, self.model.parameters(),retain_graph=True)

		meta_grads = {name:g for ((name, _), g) in zip(self.model.named_parameters(), grads)}

		hooks = []
		for (k,v) in self.model.named_parameters():
			def get_closure():
				key = k
				def replace_grad(grad):
					return meta_grads[key]
				return replace_grad
			hooks.append(v.register_hook(get_closure()))
		
		self.optimizer.zero_grad()
		self.loss.backward()
		self.optimizer.step()

		for h in hooks:
			h.remove()

		if j%50==0 and i==0:

			state = {
				'epoch': j,
				'state_dict': self.model.state_dict(),
				'optimizer': self.optimizer.state_dict(),
			}
			str_path = model_path + '/model_auto' + '.h5'
			torch.save(state,str_path)
			print("Saving the model")

		return self.loss.data[0]


	def train_maml(self,meta_train_noisy,meta_train_clean,train_datapts,meta_train_datapts,num_iter):

		num_tasks,num_data,num_features = meta_train_noisy.shape

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

				noisy = meta_train_noisy[t,idx_train,:]
				clean = meta_train_clean[t,idx_train,:]

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
			for t in range(num_tasks):

				#Sample K datapoints from the task t
				idx_meta = np.random.randint(num_data,size=D)

				noisy = meta_train_noisy[t,idx_meta,:]
				clean = meta_train_clean[t,idx_meta,:]

				noisy = np_to_variable(noisy, requires_grad=True)
				clean = np_to_variable(clean, requires_grad=False)

				# output2 = self.model(noisy)

				#Get the loss w.r.t the theta_i network
				self.set_weights(theta_list[t])
				approx_clean = self.model(noisy)
				self.loss_outer = self.criterion(approx_clean, clean)

				# Set the model weights to theta before training
				#Train with this theta on the D samples
				self.meta_optimizer.zero_grad()
				grads = torch.autograd.grad(self.loss_outer, self.model.parameters())
				grads = clip_grad_norm_(grads,0.5)
				#Pass the gradients directly to the Custom Adam optimizer
				self.meta_optimizer.step(grads)
			
				
				# self.set_weights(theta)
				# self.meta_optimizer.zero_grad()
				# self.loss.backward()
				# self.meta_optimizer.step()

				# Theta will now have the updated parameters
				theta = self.get_weights()

				#Add up the losses from each of these networks
				combined_loss += self.loss.data[0]

			print("Average Loss in iteration %s is %1.2f" %(i,combined_loss/num_tasks))


	
def parse_arguments():
	# Command-line flags are defined here.
	parser = argparse.ArgumentParser()
	parser.add_argument('--num-epochs', dest='num_epochs', type=int,
						default=1000, help="Number of epochs to train on.")
	parser.add_argument('--train_lr', dest='train_lr', type=float,
						default=1e-5, help="The training learning rate.")
	parser.add_argument('--meta_lr', dest='meta_lr', type=float,
						default=1e-4, help="The meta-training learning rate.")
	parser.add_argument('--batch_size', type=int,
						default=400, help="Batch size")
	parser.add_argument('--hidden_size', type=int,
						default=500, help="hidden size")
	parser.add_argument('--clean_dir', type=str, default='TIMIT/TRAIN/', metavar='N',
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
	parser.add_argument('--clean_dir_test', type=str, default='TIMIT/TEST/', metavar='N',
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
	
	ae_model = Auto(1771, 161)
	if torch.cuda.is_available():
		ae_model.cuda()

	ae_model.train()

	# Create plot
	fig1 = plt.figure()
	ax1 = fig1.gca()
	ax1.set_title('Loss vs Epochs')

	# #one data loader for each SNR
	# meta_training_data_1 = LoadData(tsv_file=meta_training_file, clean_dir=clean_dir,frame_size = frame_size,SNR=-6,noise=noise_type)
	# meta_training_data_2 = LoadData(tsv_file=meta_training_file, clean_dir=clean_dir,frame_size = frame_size,SNR=-3,noise=noise_type)
	# meta_training_data_3 = LoadData(tsv_file=meta_training_file, clean_dir=clean_dir,frame_size = frame_size,SNR=0,noise=noise_type)
	# meta_training_data_4 = LoadData(tsv_file=meta_training_file, clean_dir=clean_dir,frame_size = frame_size,SNR=3,noise=noise_type)
	# meta_training_data_5 = LoadData(tsv_file=meta_training_file, clean_dir=clean_dir,frame_size = frame_size,SNR=6,noise=noise_type)

	# reg_training_data = LoadData(tsv_file=reg_training_file,clean_dir=clean_dir,frame_size = frame_size,noise=noise_type)

	# #ACTUAL DATA LOADERS for each meta/reg

	# meta_train_loader_1 = DataLoader(meta_training_data_1,batch_size=batch_size,shuffle=True,num_workers=0)
	# meta_train_loader_2 = DataLoader(meta_training_data_2,batch_size=batch_size,shuffle=True,num_workers=0)
	# meta_train_loader_3 = DataLoader(meta_training_data_3,batch_size=batch_size,shuffle=True,num_workers=0)
	# meta_train_loader_4 = DataLoader(meta_training_data_4,batch_size=batch_size,shuffle=True,num_workers=0)
	# meta_train_loader_5 = DataLoader(meta_training_data_5,batch_size=batch_size,shuffle=True,num_workers=0)

	# reg_train_loader = DataLoader(reg_training_data,batch_size=batch_size,shuffle=True,num_workers=0)
	
	# noisy_data1 = np.load('spectograms_train/noise/train/noise_-6.npy')
	# noisy_data2 = np.load('spectograms_train/noise/train/noise_-3.npy')
	# noisy_data3 = np.load('spectograms_train/noise/train/noise_0.npy')
	# noisy_data4 = np.load('spectograms_train/noise/train/noise_3.npy')
	# noisy_data5 = np.load('spectograms_train/noise/train/noise_6.npy')

	# clean_data = np.load('spectograms_train/clean/train/clean_single.npy')
	

	# noisy_sq1 = np.reshape(noisy_data1,[noisy_data1.shape[0]*noisy_data1.shape[1],noisy_data1.shape[2]])
	# noisy_sq2 = np.reshape(noisy_data2,[noisy_data2.shape[0]*noisy_data2.shape[1],noisy_data2.shape[2]])
	# noisy_sq3 = np.reshape(noisy_data3,[noisy_data3.shape[0]*noisy_data3.shape[1],noisy_data3.shape[2]])
	# noisy_sq4 = np.reshape(noisy_data4,[noisy_data4.shape[0]*noisy_data4.shape[1],noisy_data4.shape[2]])
	# noisy_sq5 = np.reshape(noisy_data5,[noisy_data5.shape[0]*noisy_data5.shape[1],noisy_data5.shape[2]])

	# noisy_total = []

	# noisy_total.append(noisy_sq1)
	# noisy_total.append(noisy_sq2)
	# noisy_total.append(noisy_sq3)
	# noisy_total.append(noisy_sq4)
	# noisy_total.append(noisy_sq5)
	
	# noisy_total = np.array(noisy_total)

	# meta_train_noisy = noisy_total

	# noisy_total = np.reshape(noisy_total,[noisy_total.shape[0]*noisy_total.shape[1],noisy_total.shape[2]])
	

	# clean_sq1 = np.reshape(clean_data,[clean_data.shape[0]*clean_data.shape[1],clean_data.shape[2]])
	# clean_sq2 = np.reshape(clean_data,[clean_data.shape[0]*clean_data.shape[1],clean_data.shape[2]])
	# clean_sq3 = np.reshape(clean_data,[clean_data.shape[0]*clean_data.shape[1],clean_data.shape[2]])
	# clean_sq4 = np.reshape(clean_data,[clean_data.shape[0]*clean_data.shape[1],clean_data.shape[2]])
	# clean_sq5 = np.reshape(clean_data,[clean_data.shape[0]*clean_data.shape[1],clean_data.shape[2]])

	# clean_total =[]

	# clean_total.append(clean_sq1)
	# clean_total.append(clean_sq2)
	# clean_total.append(clean_sq3)
	# clean_total.append(clean_sq4)
	# clean_total.append(clean_sq5)

	# clean_total = np.array(clean_total)

	# meta_train_clean = clean_total

	# clean_total = np.reshape(clean_total,[clean_total.shape[0]*clean_total.shape[1],clean_total.shape[2]])

	# shuffle_idx = np.random.permutation(noisy_total.shape[0])

	# noisy_total = noisy_total[shuffle_idx]
	# clean_total = clean_total[shuffle_idx]

	# print(meta_train_noisy.shape)
	# print(meta_train_clean.shape)

	
	dae = Denoise(ae_model,train_lr,meta_lr)

	path_name = './figures/meta_train_plots'
	str_path1 = 'training_loss_mask_normal_total.png'
	plot1_name = os.path.join(path_name,str_path1)

	model_path = 'models/meta/'

	if not os.path.exists(path_name):
		os.makedirs(path_name)

	if not os.path.exists(model_path):
		os.makedirs(model_path)

	# Normal training with one SNR

	noisy_total = np.ones([2500,1771])
	clean_total = np.ones([2500,161])

	num_samples = int(noisy_total.shape[0])
	for j in range(num_epochs):
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
	  print('epoch [{}/{}], MSE_loss:{:.4f}'.format(j + 1, num_epochs, total_loss))
	  ax1.scatter(j+1, total_loss)
	  if j%100 == 0:
		  ax1.figure.savefig(plot1_name)
	
	meta_train_noisy = np.ones([5,4610,1771])
	meta_train_clean = np.ones([5,4610,161])
	train_datapts = 500
	meta_train_datapts = 500
	num_iter = 10000        
	#Meta-training with five SNR
	# dae.train_maml(meta_train_noisy,meta_train_clean,train_datapts,meta_train_datapts,num_iter)




if __name__ == '__main__':
	main(sys.argv)

