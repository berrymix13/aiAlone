# 필요 패키지 로드
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import ToTensor
from torchvision.transforms import Resize
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
  
from .tools import cifar_mean, cifar_std

# data를 불러오고 처리하는 함수
def getTransform(args): 
    # 정규화에 사용될 mean과 std 불러옴
    mean = cifar_mean
    std = cifar_std

    ## transform 정의
    # compose를 이용해 두 개 이상의 수행을 묶어준다.
    transform = Compose([
        # 이미지 사이즈를 재정의
        Resize((args.img_size, args.img_size)), 
        # 기존 numpy이미지 -> Tensor로 변경
        ToTensor(),
        # 평균과 표준편자를 이용해 정규화 진행 
        Normalize(mean, std)
    ])
    return transform

def getDataLoader(args): 
    # 데이터를 불러올 때 적용시킬 transform 불러옴
    transform = getTransform(args)

    # 데이터 불러오기 
    train_dataset = CIFAR10(root='./cifar', train=True, transform=transform, download=True)
    test_dataset = CIFAR10(root='./cifar', train=False, transform=transform, download=True)

    # dataloader 
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    # dataLoader 반환
    return train_loader, test_loader

# model_type에 따라 실행될 모델을 반환하는 함수 
def getTargetModel(args): 
    # MLP
    if args.model_type == 'mlp': 
        from networks.MLP import myMLP
        model = myMLP(args.hidden_size, args.num_classes).to(args.device)
    # lenet
    elif args.model_type == 'lenet': 
        from networks.LeNet import mylenet
        model = mylenet(args.num_classes).to(args.device) 
    # lenet_linear
    elif args.model_type == 'linear': 
        from networks.LeNet import mylenet_Linear
        model = mylenet_Linear(args.num_classes).to(args.device) 
    # lenet_conv
    elif args.model_type == 'conv': 
        from networks.LeNet import mylenet_convs
        model = mylenet_convs(args.num_classes).to(args.device) 
    # lenet_incep
    elif args.model_type == 'incep': 
        from networks.LeNet import mylenet_incep
        model = mylenet_incep(args.num_classes).to(args.device) 
    # VGG
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
    # ResNet
    elif args.model_type == 'resnet': 
        if args.res_config == '18' : 
            from networks.ResNet import ResNet
            model = ResNet().to(args.device) 
        # elif args.res_config == 'b' : 
        #     from networks.VGG import VGG_B
        #     model = VGG_B(args.num_classes).to(args.device) 
        # elif args.res_config == 'c' : 
        #     from networks.VGG import VGG_C
        #     model = VGG_C(args.num_classes).to(args.device) 
        # elif args.res_config == 'd' : 
        #     from networks.VGG import VGG_D
        #     model = VGG_D(args.num_classes).to(args.device) 
        # elif args.res_config == 'e' : 
        #     from networks.VGG import VGG_E
        #     model = VGG_E(args.num_classes).to(args.device) 
    # 이외에 
    else : 
        # 일치하는 모델이 없을 경우 ValueError 띄움
        raise ValueError('no model implemented~')
    
    return model 