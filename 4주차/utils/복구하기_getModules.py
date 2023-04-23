from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import Resize
from torchvision.transforms import Normalize
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader 

from utils.복구하기_tools import cifar_mean, cifar_std

# getDataLoader로부터 transform 분리
def getTransform(args):
    transform = Compose([
    Resize((args.img_size, args.img_size)), 
    ToTensor(),
    Normalize(cifar_mean, cifar_std)
    ])
    return transform

# DataLoader
def getDataLoader(args):

    transform = getTransform(args)

    train_dataset = CIFAR10(root='./cifar', train=True, transform=transform, download=True)
    test_dataset = CIFAR10(root='./cifar', train=False, transform=transform, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, test_loader

# TargetModel
def getTargetModel(args):
    if args.model_type == 'mlp':
        from networks.MLP import myMLP
        model = myMLP(args.hidden_size, args.num_classes)

    elif args.model_type =='lenet':
        from networks.LeNet import mylenet
        model = mylenet(args.num_classes).to(args.device)

    elif args.model_type =='linear':
        from networks.LeNet import mylenet_Linear
        model = mylenet_Linear(args.num_classes).to(args.device)

    elif args.model_type =='conv':
        from networks.LeNet import mylenet_convs
        model = mylenet_convs(args.num_classes).to(args.device)

    elif args.model_type =='incep':
        from networks.LeNet import myLeNet_incep
        model = myLeNet_incep(args.num_classes).to(args.device)
    
    else:
        raise ValueError('no model implemented~')

    return model