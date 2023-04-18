from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import Resize
from torchvision.transforms import Normalize
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader 

from .tools import cifar_mean, cifar_std

def getTransform(args):
    transform = Compose([
    Resize((args.img_size, args.img_size)), 
    ToTensor(),
    Normalize(cifar_mean, cifar_std)
    ])
    return transform

def getDataLoader(args):

    transform = getTransform(args)

    train_dataset = CIFAR10(root='./cifar', train=True, transform=transform, download=True)
    test_dataset = CIFAR10(root='./cifar', train=False, transform=transform, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, test_loader

def getTargetModel(args):
    if args.model_type == 'mlp':
        # ..은 파일 하나 위, ...은 파일 하나 더 위!
        # error가 많이 남! -> sys를 import 해서 경로를 강제로 지정!
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

    # elif args.model_type == 'vgg': 
    #     if args.vgg_type == 'a' : 
    #         from networks.VGG import VGG_A
    #         model = VGG_A(args.num_classes).to(args.device) 
    #     elif args.vgg_type == 'b' : 
    #         from networks.VGG import VGG_B
    #         model = VGG_B(args.num_classes).to(args.device) 
    #     elif args.vgg_type == 'c' : 
    #         from networks.VGG import VGG_C
    #         model = VGG_C(args.num_classes).to(args.device) 
    #     elif args.vgg_type == 'd' : 
    #         from networks.VGG import VGG_D
    #         model = VGG_D(args.num_classes).to(args.device) 
    #     elif args.vgg_type == 'e' : 
    #         from networks.VGG import VGG_E
    #         model = VGG_E(args.num_classes).to(args.device) 
    
    else:
        raise ValueError('no model implemented~')

    return model