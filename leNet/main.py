import os.path
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms, datasets
import torch.utils.data as torch_data

# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'frog', 'horse', 'ship', 'truck')

Loss = []
Accuracy = []

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.maxPool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.maxPool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = F.log_softmax(self.fc3(x), dim=1)
        return output


def train_model(model, criterion, train_loader, optimizer, epoch):
    model.train()
    total = 0
    correct = 0.0
    running_loss = 0.0
    for _ in range(epoch):
        # enumerate
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total += labels.size(0)
            correct += (output.argmax(dim=1) == labels).sum().item()
            if i % 100 == 99:
                print('[Epoch: %d, Batch: %5d] Loss: %.3f' % (_ + 1, i + 1, running_loss / 1000))
                print("accuracy: {:.6f}%".format(100 * (correct / total)))
                running_loss = 0.0
        Loss.append(running_loss / len(train_loader))  # 平均损失
        Accuracy.append(correct / total)  # 平均准确率

    print('Finished')


def test_model(model, test_loader):
    print("start test:")
    # 模型开启验证的模式
    model.eval()
    correct = 0.0
    test_loss = 0.0
    total = 0
    with torch.no_grad():
        for data, label in test_loader:
            output = model.forward(data)
            loss = criterion(output, label)
            test_loss += loss
            res = output.argmax(dim=1)
            total += label.size(0)
            correct += (res == label).sum().item()

        print("test_avarage_loss: {:.6f}, accuracy: {:.6f}%".format(test_loss / total, 100 * (correct / total)))


# 如果有参数加载参数
def load_param(model, path):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))


# 保留跑出来的参数
def save_param(model, path):
    torch.save(model.state_dict(), path)


if __name__ == '__main__':
    # load data_set
    pipline_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    pipline_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = datasets.CIFAR10('data', train = True, download=True, transform=pipline_train)
    test_set = datasets.CIFAR10('data', train = False, download=True, transform=pipline_test)

    train_loader = torch_data.DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = torch_data.DataLoader(test_set, batch_size=32, shuffle=True)

    # 选择device, 没设cuda，硬跑
    # device = torch.device("cpu")
    leNet_model = LeNet()

    # 定义交叉熵函数
    criterion = nn.CrossEntropyLoss()
    # 优化器为随机梯度下降
    optimizer = optim.SGD(leNet_model.parameters(), lr=0.001, momentum=0.9)
    load_param(leNet_model, 'model_pkl')
    train_model(leNet_model, criterion, train_loader, optimizer, 3)
    save_param(leNet_model, 'model_pkl')
    plt.subplot(2, 1, 1)
    plt.plot(range(1, len(Loss) + 1), Loss)
    plt.title('Loss')

    plt.subplot(2, 1, 2)
    plt.plot(range(1, len(Accuracy) + 1), Accuracy)
    plt.title('Accuracy')

    plt.tight_layout()  # 调整子图布局
    plt.show()

    test_model(leNet_model, test_loader)
