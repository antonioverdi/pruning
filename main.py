import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnet
import json
import prune
import utils


model_names = sorted(name for name in resnet.__dict__
	if name.islower() and not name.startswith("__")
					 and name.startswith("resnet")
					 and callable(resnet.__dict__[name]))

parser = argparse.ArgumentParser(description='ResNet56 pruning experiment properties')
parser.add_argument('--arch', '-a', metavar='ARCH', default='ResNet50',
					choices=model_names,
					help='model architecture: ' + ' | '.join(model_names) +
					' (default: resnet56)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
					help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
					help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
					help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
					metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
					metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
					help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
					metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
					metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
					help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
					help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
					help='use pre-trained model')
parser.add_argument('--save-dir', dest='save_dir',
					help='The directory used to save the trained models',
					default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
					help='Saves checkpoints at every specified number of epochs',
					type=int, default=10)
parser.add_argument('--prune_amount', dest='prune_amount', help='Amount to prune per epoch',
					default=0.2)
parser.add_argument('--prune_epochs', dest='prune_epochs', help='Amount to prune per epoch',
					default=4)
parser.add_argument('--prune_smallest', action="store_true", help="Prunes weights with the smallest change between epochs")
parser.add_argument('--prune_greatest', action="store_true", help="Prunes weights with the greatest change between epochs")
parser.add_argument('--cpu', action="store_true", help="set true to avoid moving model to Cuda")
best_prec1 = 0


def main():
	global args, best_prec1
	args = parser.parse_args()

	# Check the save_dir exists or not
	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)

	model = torch.nn.DataParallel(resnet.__dict__[args.arch]())
	previous_epoch = torch.nn.DataParallel(resnet.__dict__[args.arch]())
	
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
									 std=[0.229, 0.224, 0.225])

	train_loader = torch.utils.data.DataLoader(
		datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
			transforms.RandomHorizontalFlip(),
			transforms.RandomCrop(32, 4),
			transforms.ToTensor(),
			normalize,
		]), download=True),
		batch_size=args.batch_size, shuffle=True,
		num_workers=args.workers, pin_memory=(not args.cpu))

	val_loader = torch.utils.data.DataLoader(
		datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
			transforms.ToTensor(),
			normalize,
		])),
		batch_size=128, shuffle=False,
		num_workers=args.workers, pin_memory=(not args.cpu))

	if not args.cpu:
		model.cuda()
		previous_epoch.cuda()

	# Resume from a checkpoint if desired
	if args.resume:
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			checkpoint = torch.load(args.resume)
			args.start_epoch = checkpoint['epoch']
			best_prec1 = checkpoint['best_prec1']
			model.load_state_dict(checkpoint['state_dict'])
			print("=> loaded checkpoint '{}' (epoch {})"
				  .format(args.evaluate, checkpoint['epoch']))
		else:
			print("=> no checkpoint found at '{}'".format(args.resume))

	if not args.cpu:
		cudnn.benchmark = True

	## Optimizer and LR scheduler
	criterion = nn.CrossEntropyLoss()
	if not args.cpu:
		criterion = nn.CrossEntropyLoss().cuda()
	optimizer = optim.SGD(model.parameters(), lr=args.lr,
					  momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

	lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[91, 136], gamma=0.1)

	if args.evaluate:
		validate(val_loader, model, criterion, args.cpu)
		return

	accuracy_dict = {}

	for epoch in range(args.start_epoch, args.epochs):

		# Train for a single epoch
		print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
		begin_time = time.time()
		train(train_loader, model, criterion, optimizer, epoch, args.cpu)
		lr_scheduler.step()

		# Evaluate on validation set
		prec1 = validate(val_loader, model, criterion, args.cpu)

		# Save if epoch is a checkpoint or if precision is best thus far
		is_best = prec1 > best_prec1
		best_prec1 = max(prec1, best_prec1)

		if epoch > 0 and epoch % args.save_every == 0:
			save_checkpoint({
				'epoch': epoch + 1,
				'state_dict': model.state_dict(),
				'best_prec1': best_prec1,
			}, filename=os.path.join(args.save_dir, 'checkpoint.th'))

		if is_best:
			save_checkpoint({
				'epoch': epoch + 1,
				'state_dict': model.state_dict(),
				'best_prec1': best_prec1,
			}, filename=os.path.join(args.save_dir, 'model.th'))
		print("Epoch Total Time: {:.3f}".format(time.time() - begin_time))

		accuracy_dict['epoch{}'.format(epoch)] = prec1

		# If we are pruning the smallest at each epoch
		if args.prune_smallest and epoch>0 and epoch<args.prune_epochs:
			model = prune.prune_smallest(previous_epoch, model, args.prune_amount, epoch)
		
		# If we are pruning the greatest at each epoch
		if args.prune_greatest and epoch>0 and epoch<args.prune_epochs:
			model = prune.prune_greatest(previous_epoch, model, args.prune_amount)
 
		# Save the state dict so we can compare it with the state dict from the next epoch
		previous_epoch.load_state_dict(model.state_dict())
	
		# Calculate and print out sparsity at the end of each epoch
		sparsity_per_layer = utils.calculate_sparsity(model)
		global_sparsity = sum(sparsity_per_layer)/len(sparsity_per_layer)
		print("Global Sparsity: " + global_sparsity)


	# Once the model has finished training, save the accuracies per epoch to a JSON file
	acc_file = open("accuracies.json", "w")
	json.dump(accuracy_dict, acc_file)
	acc_file.close()


def train(train_loader, model, criterion, optimizer, epoch, cpu=False):
	"""
	Run one train epoch
	"""
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()

	# Switch to train mode
	model.train()

	end = time.time()
	for i, (input, target) in enumerate(train_loader):

		# measure data loading time
		data_time.update(time.time() - end)


		if cpu:
			input_var = input
		else:
			target = target.cuda()
			input_var = input.cuda()

		target_var = target

		# compute output
		output = model(input_var)
		loss = criterion(output, target_var)

		# Compute grad and perform SGD
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		output = output.float()
		loss = loss.float()

		# measure accuracy and record loss
		prec1 = accuracy(output.data, target)[0]
		losses.update(loss.item(), input.size(0))
		top1.update(prec1.item(), input.size(0))

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % args.print_freq == 0:
			print('Epoch: [{0}][{1}/{2}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
					  epoch, i, len(train_loader), batch_time=batch_time,
					  data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion, cpu=False):
	"""
	Run evaluation
	"""
	batch_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()

	# switch to evaluate mode
	model.eval()

	end = time.time()
	with torch.no_grad():
		for i, (input, target) in enumerate(val_loader):
			if not cpu:
				target = target.cuda()
				input_var = input.cuda()
				target_var = target.cuda()
			else:
				input_var = input
				target_var = target

			# compute output
			output = model(input_var)
			loss = criterion(output, target_var)

			output = output.float()
			loss = loss.float()

			# measure accuracy and record loss
			prec1 = accuracy(output.data, target)[0]
			losses.update(loss.item(), input.size(0))
			top1.update(prec1.item(), input.size(0))

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if i % args.print_freq == 0:
				print('Test: [{0}/{1}]\t'
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
						  i, len(val_loader), batch_time=batch_time, loss=losses,
						  top1=top1))

	print(' * Prec@1 {top1.avg:.3f}'
		  .format(top1=top1))

	return top1.avg

def save_checkpoint(state, filename='checkpoint.pth.tar'):
	"""
	Save the training model
	"""
	torch.save(state, filename)

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res

if __name__ == '__main__':
	main()
