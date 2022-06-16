import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import glob
#from torchvision.datasets import ImageFolder

from torch.autograd import Variable as V

import cv2
import os
import numpy as np
from pathlib import Path
import pathlib

from time import time
from datetime import datetime

from networks.unet import Unet
from networks.dunet import Dunet
from networks.dinknet import LinkNet34, DinkNet34, DinkNet50, DinkNet101, DinkNet34_less_pool
from framework import MyFrame
from loss import dice_bce_loss
from data import ImageFolder

SHAPE = (1024,1024)
#current_dir = 
data_dir = Path('dataset/train/')
#print('data_dir',data_dir)
imagelist = filter(lambda x: x.find('sat')!=-1, os.listdir(str(data_dir)))
#print('imagelist',imagelist[0])
#imagelist = glob.glob('dataset/train/*sat.jpg')
#print('imagelist',imagelist)

trainlist = list(map(lambda x: x[:-8], imagelist))
#print('trainlist',list(trainlist))

NAME = 'log01_dink34'
BATCHSIZE_PER_CARD = 4

solver = MyFrame(DinkNet34, dice_bce_loss, 2e-4)
#batchsize = int(torch.cuda.device_count()/2) * BATCHSIZE_PER_CARD
batchsize=16
#print('torch.cuda.device_count(): ',torch.cuda.device_count())

dataset = ImageFolder(trainlist, str(data_dir))

#print("dataset: ",dataset)
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batchsize,
    shuffle=True,
    num_workers=2)

tic = time()
no_optim = 0
total_epoch = 300
train_epoch_best_loss = 100.

# Create a numer of empty arrays and variables for updating during training.
train_loss_array = np.array([])

for epoch in range(1, total_epoch + 1):
    data_loader_iter = iter(data_loader)
    train_epoch_loss = 0
    print('start epoch {}'.format(epoch))
    for img, mask in data_loader_iter:
        solver.set_input(img, mask)
        train_loss = solver.optimize()
        train_epoch_loss += train_loss
    train_epoch_loss /= len(data_loader_iter)   
    print('train_epoch_loss: ',train_epoch_loss)
    train_cost_array = np.append(train_loss_array, train_epoch_loss)
    epochs = np.arange(epoch)

        
    if train_epoch_loss >= train_epoch_best_loss:
        no_optim += 1
    else:
        no_optim = 0
        train_epoch_best_loss = train_epoch_loss
        #solver.save('weights/'+NAME+'.th')
        solver.save('weights/'+NAME+'.pt')
    if no_optim > 6:
        print('early stop at {} epoch'.format(epoch))
        break
    if no_optim > 3:
        if solver.old_lr < 5e-7:
            break
        #solver.load('weights/'+NAME+'.th')
        solver.load('weights/'+NAME+'.th')
        solver.update_lr(5.0, factor = True)
        
uid = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p") 
mlflow.set_experiment('road test' + uid )

mlflow.log_param("model_name", "D-LinkNet34")
mlflow.log_param("dataset_name", "DeepGlobeRoad2019")
mlflow.log_param("num_epochs", epochs)
mlflow.log_param("optimizer_name", "ADAM")
mlflow.log_param("loss_func_name", "BCEloss")
mlflow.log_param("init learn_rate", "2e-4")
mlflow.log_param("input_dims", SHAPE)
mlflow.log_param("batch_size", batchsize)
mlflow.setbatchsize

#plot the error of each class per epoch for training and validation
epochs = np.arange(len(training_cost))
plt.figure(1)
plt.plot(epochs, training_cost, 'b-')
plt.plot(epochs, validation_cost, 'r-')
plt.plot(epochs, error_rates, 'g-')
plt.legend(['Training Cost', 'Validation Cost', 'Error Rate'], loc='upper right')
plt.xlabel('Epoch Number')
plt.ylabel('Cost Function Value & Error Rate')
plt.savefig(save_path[:-4] + ".png")
mlflow.log_artifact(save_path[:-4] + ".png")
plt.close()

print('Finish!')

