# %%
import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms, datasets


##1. 데이터를 불러오는 부분에서 training 을 false로 지정한다
##2.

# %% 1. 트레이닝에 필요한 하이퍼 파라미터 설정
lr = 1e-3
batch_size = 64
num_epochs = 10

ckpt_dir = './checkpoint'
log_dir = 'log'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% 2. 네트워크 구축하기
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #레이어 정의
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, stride=1, padding=0, bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, stride=1, padding=0, bias=True)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.relu2 = nn.ReLU()

        self.fc1 = nn.Linear(320, 50, bias=True)
        self.relu1_fc1 = nn.ReLU()
        self.drop1_fc1 = nn.Dropout2d(p=0.5)

        self.fc2 = nn.Linear(50, 10, bias=True)

    def forward(self, x):
        #레이어 호출
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.drop2(x)
        x = self.pool2(x)
        x = self.relu2(x)

        x = x.view(-1, 320)

        x = self.fc1(x)
        x = self.relu1_fc1(x)
        x = self.drop1_fc1(x)

        x = self.fc2(x)

        return x

# %% 3. 네트워크를 저장하거나 불러오는 함수 작성하기
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net' : net.state_dict(), 'optim' : optim.state_dict()},
                   './%s/model_epoch%d.pth' % (ckpt_dir, epoch))

def load(ckpt_dir, net, optim):
    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort()

    dict_model = torch.load('./%s/%s' % (ckpt_dir, ckpt_lst[-1]))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])

    return net, optim

# %% 4. mnist 데이터 불러오기
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST('./', train=False, download=True, transform=transforms)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

num_data = len(loader.dataset)
num_batches = np.ceil(num_data / batch_size)

# %% 5. 네트워크 설정 및 필요한 손실함수 구현하기
net = Net().to(device)
params = net.parameters()

fn_loss = nn.CrossEntropyLoss().to(device)
fn_pred = lambda x: torch.softmax(x, dim=1)
fn_acc = lambda pred, label : ((pred.max(dim=1)[1] == label).type(torch.float)).mean()

optim = torch.optim.Adam(params, lr=lr)

writer = SummaryWriter(log_dir=log_dir)

# eval() 신규 작성
net, optim = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

# %% eval() 에서 변경하는 부분
#for epoch in range(1, num_epochs+1):
with torch.no_grad():
    #net.train()
    net.eval()

    loss_arr = []
    acc_arr = []

    for batch, (input, labels) in enumerate(loader, 1):
        input = input.to(device)
        label = labels.to(device)

        outputs = net(input)
        preds = fn_pred(outputs)

        #optim.zero_grad()

        loss = fn_loss(preds, label)
        acc = fn_acc(preds, label)

        #loss.backward()

        #optim.step()

        loss_arr += [loss.item()]
        acc_arr += [acc.item()]

        print("train : batch %04d/%04d | loss : %.4f | acc %.4f" %
              ( batch, num_batches, np.mean(loss_arr), np.mean(acc_arr)))






