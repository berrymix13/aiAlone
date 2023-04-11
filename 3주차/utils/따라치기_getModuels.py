from mics.따라치기_getCIFAR_std_mean import cifar_mean, cifar_std
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
from torch.utils.data import DataLoader

def getDataLoader(args):
    transform = Compose([
            Resize((args.img_size, args.img_size)),
            ToTensor(),
            Normalize(cifar_mean, cifar_std)
        ])
        
    train_dataset = CIFAR10(root='./cifat', train=True, transform=transform, download=True)
    test_dataset = CIFAR10(root='./cifat', train=False, transform=transform, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, test_loader

def getTargetModel(args):
    if args.model_type == 'mlp':
        from networks.따라치기_MLP import myMLP
        model = myMLP(args.hidden_size, args.num_classes).to(args.device)
    
    elif args.model_type == 'lenet':
        from networks.따라치기_LeNet import mylenet
        model = mylenet(args.num_classes).to(args.device)
    
    elif args.model_type == 'linear':
        from networks.따라치기_LeNet import mylenet_linear
        model = mylenet_linear(args.num_classes).to(args.device)
   
    elif args.model_type == 'convs':
        from networks.따라치기_LeNet import mylenet_convs
        model = mylenet_convs(args.num_classes).to(args.device)
   
    elif args.model_type == 'incep':
        from networks.따라치기_LeNet import mylenet_incep
        model = mylenet_incep(args.num_classes).to(args.device)
   
    else:
        raise ValueError('no model implemented!')
    
    return model
