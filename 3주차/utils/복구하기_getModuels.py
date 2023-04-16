from mics.따라치기_getCIFAR_std_mean import cifar_mean, cifar_std
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
from torch.utils.data import DataLoader

# data를 불러오고 처리하는 함수
def getDataLoader(args):
    # args에 포함되어 있는 변수들은 args.을 붙여서 사용한다
    transform = Compose([
        Resize((args.img_size, args.img_size)),
        ToTensor(),
        Normalize(cifar_mean, cifar_std)
    ])

    train_data = CIFAR10(root='./cifar', train=True, transform=transform, download=True)
    test_data = CIFAR10(root='./cifar', train=False, transform=transform, download=True)

    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size)

    return train_loader, test_loader

# model_type에 따라 실행될 모델을 반환하는 함수 
def getTargetModel(args):
    # 각 모델을 지정할 때는 networks로부터 불러오기 (mlp, lenet, linear, convs, incep)
    if args.model_type == 'mlp':
        from networks.복구하기_MLP import myMLP
        model = myMLP(args.hidden_size, args.num_classes).to(args.device)

    elif args.model_type == 'lenet':
        from networks.복구하기_LeNet import mylenet
        model = mylenet(args.num_classes).to(args.device)
    
    elif args.model_type == 'linear':
        from networks.복구하기_LeNet import mylenet_linear
        model = mylenet_linear(args.num_classes).to(args.device)
   
    elif args.model_type == 'convs':
        from networks.복구하기_LeNet import mylenet_convs
        model = mylenet_convs(args.num_classes).to(args.device)
   
    elif args.model_type == 'incep':
        from networks.복구하기_LeNet import mylenet_incep
        model = mylenet_incep(args.num_classes).to(args.device)
   
    else:
        # raise  : 에러를 발생시킴
        # 일치하는 모델이 없을 경우 ValueError 띄움
        raise ValueError('no model implemented!')
    
    return model
    # 자꾸 return을 잊어서 오류남  ## 신경쓸 것!!