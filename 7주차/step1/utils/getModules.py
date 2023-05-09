from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import ToTensor
from torchvision.transforms import Resize
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
  
from .tools import cifar_mean, cifar_std

def getTransform(args): 
    if args.fine_tuning : 
        from torchvision.transforms._presets import ImageClassification
        transform = ImageClassification(crop_size=args.img_size, resize_size=args.img_size)
    
    else :
        mean = cifar_mean
        std = cifar_std

        transform = Compose([
            Resize((args.img_size, args.img_size)), 
            ToTensor(),
            Normalize(mean, std)
        ])
    return transform

def getDataLoader(args): 
    
    transform = getTransform(args)
    
    if args.data == 'cifar':
        train_dataset = CIFAR10(root='./cifar', train=True, transform=transform, download=True)
        test_dataset = CIFAR10(root='./cifar', train=False, transform=transform, download=True)

    else:
        if args.dataset == 'imagefolder':
            from torchvision.datasets import ImageFolder
            train_dataset = ImageFolder(root='/home/dataset/dog_v1_TT/train', transform=transform)
            test_dataset = ImageFolder(root='/home/dataset/dog_v1_TT/test', transform=transform)

        elif args.dataset == 'cust1':
            from .dogdataset import DogDataset
            train_dataset = DogDataset(root='/home/dataset/dog_v1_TT/train', transform=transform)
            test_dataset = DogDataset(root='/home/dataset/dog_v1_TT/test', transform=transform)
        
        elif args.dataset == 'cust2':
            from sklearn.model_selection import train_test_split
            tmp_dataset = DogDataset(root='/home/dataset/dog_v1', transform=transform)
            train_dataset, test_dataset = train_test_split(tmp_dataset, train_size= 0.8, random_state=2023)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, test_loader

def getTargetModel(args): 
    if args.model_type == 'mlp': 
        from networks.MLP import myMLP
        model = myMLP(args.hidden_size, args.num_classes).to(args.device)
    elif args.model_type == 'lenet': 
        from networks.LeNet import mylenet
        model = mylenet(args.num_classes).to(args.device) 
    elif args.model_type == 'linear': 
        from networks.LeNet import mylenet_Linear
        model = mylenet_Linear(args.num_classes).to(args.device) 
    elif args.model_type == 'conv': 
        from networks.LeNet import mylenet_convs
        model = mylenet_convs(args.num_classes).to(args.device) 
    elif args.model_type == 'incep': 
        from networks.LeNet import mylenet_incep
        model = mylenet_incep(args.num_classes).to(args.device) 
    elif args.model_type == 'vgg': 
        if args.vgg_type == 'a' : 
            from networks.VGG import VGG_A
            model = VGG_A(args.num_classes).to(args.device) 
        elif args.vgg_type == 'b' : 
            from networks.VGG import VGG_B
            model = VGG_B(args.num_classes).to(args.device) 
        elif args.vgg_type == 'c' : 
            from networks.VGG import VGG_C
            model = VGG_C(args.num_classes).to(args.device) 
        elif args.vgg_type == 'd' : 
            from networks.VGG import VGG_D
            model = VGG_D(args.num_classes).to(args.device) 
        elif args.vgg_type == 'e' : 
            from networks.VGG import VGG_E
            model = VGG_E(args.num_classes).to(args.device) 
    elif args.model_type == 'resnet': 
        if args.fine_tuning :
            print("fine_tuning", {args.fine_tuning})
            from torchvision.models import resnet18
            from torchvision.models import ResNet18_Weights
            import torch.nn as nn
            weight = ResNet18_Weights
            model = resnet18(weight, progress=True) #  그냥 여기서 끝나면 아예 그대로 쓰는것일 뿐
            

            model.fc = nn.Linear(512, 5)
            model = model.to(args.device)
       
        else : 
            from networks.ResNet import ResNet
            model = ResNet(args).to(args.device) 

    else : 
        raise ValueError('no model implemented~')
    
    return model 