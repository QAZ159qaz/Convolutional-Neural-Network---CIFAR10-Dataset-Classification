import torch
from torch import nn
import torchvision
import cv2     #opencv安装命令：pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python
import torch.nn.functional as F

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


#加载模型
model = torch.load("mynet_0.pth") 
print(model)

#分类名称
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model.eval()  # 把模型转为test模式

with torch.no_grad():   #测试无需梯度回传，降低计算量
    for k in ('horse.jpg','plane.jpg','trunk.jpg','ship.jpg','cat.jpg'):
        img = cv2.imread(k)  # 读取要预测的图片
        #转换为tensor数据类型
        trans = torchvision.transforms.Compose(
                            [
                             torchvision.transforms.ToTensor(),
                             torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                            ])

        img = trans(img)
        # print(img.size())
        img = img.unsqueeze(0)  # 图片扩展多一维,因为输入到保存的模型中是4维的[batch_size,通道,长，宽]，而普通图片只有三维，[通道,长，宽]
        # 扩展后，为[1，1，32，32]
        # print(img.size())
        output = model(img)
        prob = F.softmax(output, dim=1)  # prob是10个分类的概率
        # print(prob)
        value, predicted = torch.max(output.data, 1)
        # print(predicted.item())
        pred_class = classes[predicted.item()]
        print('原始类型：',k[:-4], '预测结果：',pred_class)

