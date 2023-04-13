from torchvision.datasets import CIFAR10

train_dataset = CIFAR10(root='./cifar', train = True, download=True)
cifar_mean = train_dataset.data.mean(axis=(0, 1, 2)) / 255.0
cifar_std = train_dataset.data.std(axis=(0, 1, 2)) / 255.0
