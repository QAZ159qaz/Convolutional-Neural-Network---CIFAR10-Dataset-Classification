import numpy
import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset,DataLoader

# 下载数据集cifar10

train_data = torchvision.datasets.CIFAR10('./DATA', train=True, transform=torchvision.transforms.ToTensor(), 
                                            download=True)
test_data = torchvision.datasets.CIFAR10('./DATA', train=False, transform=torchvision.transforms.ToTensor(), 
                                            download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)
print('训练集长度：{}'.format(train_data_size))
print('测试集长度：{}'.format(test_data_size))
#加载dataloader

train_data_loader = DataLoader(train_data, batch_size=64)
test_data_loader = DataLoader(test_data, batch_size=64)

# 搭建模型

class Mynet(nn.Module):
    def __init__(self):
        super(Mynet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)

        return x
        


#实例化神经网络
net = Mynet()

#实例化损失函数
loss_fn = nn.CrossEntropyLoss()
#实例化优化函数
learning_rate = 1e-2  # 0.01
optimizer = torch.optim.SGD(net.parameters(), lr= learning_rate)

#开始训练，设置参数

total_train_step = 0

total_test_step = 0

epoch = 10
for i in range(epoch): 
    print('实验2开始第 {} 轮训练开始'.format(i+1))


    for data in train_data_loader:   #加载数据集
        imgs, targets = data
        output = net(imgs)             #将图片输入模型
        loss = loss_fn(output,targets)  #计算模型输出与标签loss

        optimizer.zero_grad()    #每个循环梯度清零
        loss.backward()          #梯度回传
        optimizer.step()         #更新参数

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print('第{}次训练开始： LOSS= {}'.format(total_train_step,loss.item()))
    
    total_loss = 0
    total_acc = 0
    with torch.no_grad():
        for data in test_data_loader:
            imgs, targets = data
            output = net(imgs)
            loss = loss_fn(output,targets)               #计算loss
            total_loss = total_loss + loss               #计算测试集的整体准确率
            acc = (output.argmax(1) == targets).sum()    #计算准确率 
            total_acc = total_acc + acc
        print('total LOSS: {}'.format(total_loss))
        print('准确率为: {}'.format(total_acc/len(test_data)))

    torch.save(net, 'mynet_{}.pth'.format(i))
    print('模型保存完成')




