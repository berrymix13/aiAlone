# 필요 패키지 불러오기
from torchvision.datasets import CIFAR10

# data 불러오기
train_dataset = CIFAR10(root='./cifar', train=True, download=True)

# data로부터 mean, std 구함 
mean = train_dataset.data.mean(axis=(0, 1, 2)) / 255.0
std = train_dataset.data.std(axis=(0, 1, 2)) / 255.0