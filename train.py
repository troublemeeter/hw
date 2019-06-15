import time
import numpy as np
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from model import ClassifyModel
from makedata import LoadData
import os
import pickle

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True


##########################
### SETTINGS
##########################

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # todo : change to multi gpu

# Hyperparameters
random_seed = 1
learning_rate = 0.1  # todo : change with step
num_epochs = 10
batch_size = 512
dropout_prob = 0.7

# Architecture
num_features =    # todo : compute
num_hidden = [128,256]
num_classes = 6

train_loader,test_loader = LoadData()

##########################
### TRAIN
##########################

torch.manual_seed(random_seed)
model = MultilayerPerceptron(num_features=num_features,
							 num_hidden = num_hidden,
                             num_classes=num_classes)

model = model.to(device)

optimizer = torch.optim.adam(model.parameters(), lr=learning_rate)

def compute_accuracy(net, data_loader):
    net.eval()
    correct_pred, num_examples = 0, 0
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.view(-1, 28*28).to(device)
            targets = targets.to(device)
            logits, probas = net(features)
            _, predicted_labels = torch.max(probas, 1)
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
        return correct_pred.float()/num_examples * 100
    

start_time = time.time()
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):
        
        features = features.to(device)
        targets = targets.to(device)
            
        ### FORWARD AND BACK PROP
        logits, probas = model(features)
        cost = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        
        cost.backward()
        
        ### UPDATE MODEL PARAMETERS
        optimizer.step()
        
        ### LOGGING
        if not batch_idx % 50:
            print ('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f' 
                   %(epoch+1, num_epochs, batch_idx, 
                     len(train_loader), cost))


    print('Epoch: %03d/%03d training accuracy: %.2f%%' % (
          epoch+1, num_epochs, 
          compute_accuracy(model, train_loader)))

    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    print('saving model ...')

    torch.save(model.state_dict(), 'model_{}.pkl'.format(epoch))  

print('ALL done. Total Training Time: %.2f min' % ((time.time() - start_time)/60))


##########################
### test
##########################
model = ClassifyModel()
model.load_state_dict(torch.load('model_{}.pkl'.format(num_epochs)))

model = model.to(device)
model.eval()

labels = []
with torch.no_grad():
    for features in test_loader:
        features = features.to(device)
        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        labels.append(str(predicted_labels))


outputfile = 'prediction'

if os.path.exists(outputfile):
	os.remove(outputfile)

if os.path.exists(outputfile+'.pkl'):
	os.remove(outputfile+'.pkl')

with open(outputfile+'.pkl','wb') as f:
	pickle.dump(labels,f)

with open(outputfile,'w',encoding='utf-8') as f:
	f.write('\n'.join(labels))
