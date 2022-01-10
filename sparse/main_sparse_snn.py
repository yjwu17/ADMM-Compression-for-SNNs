from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,sys
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import torchvision
import torchvision.transforms as transforms
from torch_prune_utility import apply_prune, get_configuration, projection, prune_weight, keep_mask
from numpy import linalg as LA
from spiking_cnn_model import *

## modify following parameters for running
data_path = r'.'
net_path = r'.'
names = 'mnist_cnn'
map_location_name = 'cpu'

admm_epoch = 30
retrain_epoch = 30
FLAGS = None

prune_configuration = get_configuration()
dense_w = {}
P1 = prune_configuration.P1
P2 = prune_configuration.P2
P3 = prune_configuration.P3
P4 = prune_configuration.P4
prune_configuration.display()

### load data
train_dataset = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

test_set = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

### creat model
model = SCNN()
model.to(device)
criterion = nn.CrossEntropyLoss()

### optimization
optimizer_train = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer_admm = torch.optim.Adam(model.parameters(), lr=1e-3)

### training or pre-model loading
# checkpoint = torch.load(net_path)
checkpoint = torch.load(net_path, map_location = map_location_name)

W_conv1 = model.conv1
W_conv2 = model.conv2
W_fc1 = model.fc1
W_fc2 = model.fc2


def evaluate_model(test_model):
    total = 0
    correct = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device)
        outputs = test_model(inputs)
        _, predicted = outputs.cpu().max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())
    print('\n\n Test Accuracy of the model on the 10000 test images: %.3f' % (100 * correct / total))


### testing

total = 0
correct = 0
evaluate_model(model)

### ADMM initialize
Z1 = W_conv1.weight.detach()
Z1 = projection(Z1, percent=P1)
U1 = torch.zeros_like(Z1)

Z2 = W_conv2.weight.detach()
Z2 = projection(Z2, percent=P2)
U2 = torch.zeros_like(Z2)
#
Z3 = W_fc1.weight.detach()
Z3 = projection(Z3, percent=P3)
U3 = torch.zeros_like(Z3)
#
Z4 = W_fc2.weight.detach()
Z4 = projection(Z4, percent=P4)
U4 = torch.zeros_like(Z4)
#
for j in range(admm_epoch):
    total = correct = 0.
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer_admm.zero_grad()
        outputs = model(inputs)

        # labels_ = torch.zeros(batch_size, 10).scatter_(1, targets.view(-1, 1), 1)
        admm_loss = F.cross_entropy(outputs, targets)  # todo 1
        reg_loss = torch.norm(model.conv1.weight) + torch.norm(model.conv2.weight) + torch.norm(
            model.fc1.weight) + torch.norm(model.fc2.weight)
        dis_loss = torch.norm(model.conv1.weight - Z1 + U1) + torch.norm(model.conv2.weight - Z2 + U2) + torch.norm(
            model.fc1.weight - Z3 + U3) + torch.norm(model.fc2.weight - Z4 + U4)
        my_loss = admm_loss + 5e-5 * reg_loss + 1e-4 * dis_loss
        my_loss.backward()
        optimizer_admm.step()

        ## monitor training results
        _, predicted = outputs.cpu().max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets.cpu()).sum().item())

    print('\n\n ADMM optimal loss :%.4f' % my_loss)
    print(' ADMM Training Accuracy of the model  : %.3f' % (
      100 * correct / total))

    ## Update Z & U

    W_conv1 = model.conv1
    W_conv2 = model.conv2
    W_fc1 = model.fc1
    W_fc2 = model.fc2

    Z1 = W_conv1.weight.detach() + U1
    Z1 = projection(Z1, percent=P1)
    U1 = U1 + W_conv1.weight.detach() - Z1

    Z2 = W_conv2.weight.detach() + U2
    Z2 = projection(Z2, percent=P2)
    U2 = U2 + W_conv2.weight.detach() - Z2

    Z3 = W_fc1.weight.detach() + U3
    Z3 = projection(Z3, percent=P3)
    U3 = U3 + W_fc1.weight.detach() - Z3

    Z4 = W_fc2.weight.detach() + U4
    Z4 = projection(Z4, percent=P4)
    U4 = U4 + W_fc2.weight.detach() - Z4

    print('Epoch: %.1d ADMM optimal loss :%.4f' % (j, my_loss))

    if j % 5 == 0:
        print(LA.norm(W_conv1.weight.detach() - Z1))
        print(LA.norm(W_conv2.weight.detach() - Z2))
        print(LA.norm(W_fc1.weight.detach() - Z3))
        print(LA.norm(W_fc2.weight.detach() - Z4))
        ## retraining process
        print(torch.sum((W_conv1.weight.detach()) != 0))

        evaluate_model(model)
        state = {
            'net': model.state_dict(),

        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/before' + names + '.t7')

### retraining

model, mask_list = apply_prune(model)
for i in range(retrain_epoch):
    for batch_idx, (images, labels) in enumerate(train_loader):
        model.zero_grad()
        optimizer_train.zero_grad()
        # images = images.float().to(device)
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        # labels_ = torch.zeros(batch_size, 10).scatter_(1, labels.view(-1, 1), 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_train.step()
        ### keeping model
        model = keep_mask(model, mask_list)
        _, predicted = outputs.cpu().max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets.cpu()).sum().item())

    evaluate_model(model)
    if i % 5 == 0:
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/after' + names + '.t7')






