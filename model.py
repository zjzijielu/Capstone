import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

class RRNN(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super().__init__()
		self.hidden_size = hidden_size
		self.i2h = nn.Linear(input_size, hidden_size)
		self.i2o = nn.Linear(input_size, output_size) 
		self.h2o = nn.Linear(hidden_size, output_size)

	def forward(self, input, hidden):
		hidden = self.i2h(input)
		h2o = self.h2o(hidden)
		i2o = self.i2o(input)
		output = h2o + i2o
		return output, hidden
	
	def initHidden(self):
		return torch.zeros(1, self.hidden_size)