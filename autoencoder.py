import sys
import argparse
import numpy as np
import os


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

def np_to_variable(x, requires_grad=False, is_cuda=True, dtype=torch.FloatTensor):
	v = Variable(torch.from_numpy(x).type(dtype), requires_grad=requires_grad)
	if is_cuda:
		v = v.cuda()
	return v


class Autoencoder(nn.Module):
	def __init__(self, input_size):
		super(Autoencoder, self).__init__()
		self.hidden_size = 100

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
	parser.add_argument('--model-config-path', dest='model_config_path',
						type=str, default='LunarLander-v2-config.json',
						help="Path to the actor model config file.")
	parser.add_argument('--num-episodes', dest='num_episodes', type=int,
						default=50000, help="Number of episodes to train on.")
	parser.add_argument('--lr', dest='lr', type=float,
						default=5e-4, help="The actor's learning rate.")
	parser.add_argument('--critic-lr', dest='critic_lr', type=float,
						default=1e-4, help="The critic's learning rate.")
	parser.add_argument('--n', dest='n', type=int,
						default=20, help="The value of N in N-step A2C.")

	# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
	parser_group = parser.add_mutually_exclusive_group(required=False)
	parser_group.add_argument('--render', dest='render',
							  action='store_true',
							  help="Whether to render the environment.")
	parser_group.add_argument('--no-render', dest='render',
							  action='store_false',
							  help="Whether to render the environment.")
	parser.set_defaults(render=False)

	return parser.parse_args()


def main(args):

	args = parse_arguments()

	num_samples = 1000
	num_features = 200

	ae_model = Autoencoder(num_features)
	ae_model.cuda()
	ae_model.train()

	num_epochs = 100 #Put the number of training samples here

	# Do the data loading here with the given matrix sizes
	data_full_noisy = np.random.rand(num_samples,num_features)
	data_full_clean = np.random.rand(num_samples,num_features)

	lr = 1e-3 # Learning rate for the model

	dae = Denoise(ae_model,lr)

	for i in range(num_epochs):

		loss = dae.train_normal(data_full_noisy,data_full_clean)

		print('epoch [{}/{}], MSE_loss:{:.4f}'
		  .format(i + 1, num_epochs, loss))




if __name__ == '__main__':
	main(sys.argv)
