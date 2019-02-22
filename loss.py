import torch
import torch.nn as nn
import numpy as np

test_add = "hello world!!!"

def main():
	N, C = 2, 3 #batch_size, num_classes
	loss = nn.NLLLoss()
	# input is of size N x C x height x width
	#m_batch = torch.randn(N, C, 4, 5)
	m_batch = torch.empty(N, C, 4, 5, dtype=torch.long).random_(0, 2)
	print (m_batch)

	#target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)
	t_batch = torch.argmax(m_batch, dim=1)
	print (t_batch)

	m1 = torch.Tensor([[
		[1,0,0,0,1],
		[1,0,0,0,1],
		[1,0,0,0,1],
		[1,0,0,0,1]
	]])
	#print (m1.size())

	m2 = torch.Tensor([[
		[0,1,0,1,0],
		[0,1,0,1,0],
		[0,1,0,1,0],
		[0,1,0,1,0]
	]])
	#print (m2.size())

	m3 = torch.Tensor([[
		[0,0,1,0,0],
		[0,0,1,0,0],
		[0,0,1,0,0],
		[0,0,1,0,0]
	]])
	#print (m3.size())

	m_batch = torch.stack((m1, m2, m3), dim=1)
	print (m_batch.size())
	"""
	t1 = torch.Tensor([[
		[0,0,0,0,0],
		[0,0,0,0,0],
		[0,0,0,0,0],
		[0,0,0,0,0]
	]])
	#print (m1.size())

	t2 = torch.Tensor([[
		[0,1,0,1,0],
		[0,1,0,1,0],
		[0,1,0,1,0],
		[0,1,0,1,0]
	]])
	#print (m2.size())

	t3 = torch.Tensor([[
		[0,0,1,0,0],
		[0,0,1,0,0],
		[0,0,1,0,0],
		[0,0,1,0,0]
	]])
	#print (m3.size())
	
	t_batch = torch.stack((t1, t2, t3), dim=0)
	print (t_batch.size())
	"""
	loss_criterion = LossMulti(num_classes=C)
	loss = loss_criterion(m_batch.float(), t_batch)

	#loss = dice_loss(m_batch.float(), t_batch.float())
	print (loss.data)

"""
In semantic segmentation we generally have a label that we want to ignore from the loss, 
this requirement is already specified by the ignore_index parameter of NLLLoss.
"""
class LossMulti:
	def __init__(self, jaccard_weight=0, class_weights=None, num_classes=1):
		if class_weights is not None:
			nll_weight = utils.cuda(
				torch.from_numpy(class_weights.astype(np.float32)))
		else:
			nll_weight = None
		self.nll_loss = nn.NLLLoss(weight=nll_weight)
		self.jaccard_weight = jaccard_weight
		self.num_classes = num_classes

	def __call__(self, outputs, targets):
		loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)
		print (self.nll_loss(outputs, targets))
		if self.jaccard_weight:
			eps = 1e-15
			for cls in range(self.num_classes):
				jaccard_target = (targets == cls).float()
				jaccard_output = outputs[:, cls].exp()
				intersection = (jaccard_output * jaccard_target).sum()

				union = jaccard_output.sum() + jaccard_target.sum()
				loss -= torch.log((intersection + eps) / (union - intersection + eps)) * self.jaccard_weight
		return loss



def dice_coeff(pred, target):
	smooth = 1.
	num = pred.size(0)
	m1 = pred.view(num, -1)  # Flatten
	m2 = target.view(num, -1)  # Flatten
	intersection = (m1 * m2).sum()
	output = (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)
	print (output)
	return output


def SoftDiceLoss(logits, targets):
	#probs = torch.sigmoid(logits)
	probs = logits
	num = targets.size(0)  #Number of batches

	score = dice_coeff(probs, targets)
	score = 1 - score.sum() / num
	
	return score

def dice_loss(input,target):
	"""
	input is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
	target is a 1-hot representation of the groundtruth, shoud have same size as the input
	"""
	assert input.size() == target.size(), "Input sizes must be equal."
	assert input.dim() == 4, "Input must be a 4D Tensor."
	uniques=np.unique(target.numpy())
	assert set(list(uniques))<=set([0,1]), "target must only contain zeros and ones"

	probs=torch.nn.functional.softmax(input, dim=0)
	num=probs*target#b,c,h,w--p*g
	num=torch.sum(num,dim=3)#b,c,h
	num=torch.sum(num,dim=2)
	

	den1=probs*probs#--p^2
	den1=torch.sum(den1,dim=3)#b,c,h
	den1=torch.sum(den1,dim=2)
	

	den2=target*target#--g^2
	den2=torch.sum(den2,dim=3)#b,c,h
	den2=torch.sum(den2,dim=2)#b,c
	

	dice=2*(num/(den1+den2))
	dice_eso=dice[:,1:]#we ignore bg dice val, and take the fg

	dice_total=-1*torch.sum(dice_eso)/dice_eso.size(0)#divide by batch_sz

	return dice_total

if __name__ == "__main__":
	main()