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


	def __init__(self):

		pass

	def train_normal(self):

		pass

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



if __name__ == '__main__':
	main(sys.argv)
