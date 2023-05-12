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
    # fine_tuning일 때 
    if args.fine_tuning : 
        # ImageClassification 패키지 로드 
        from torchvision.transforms._presets import ImageClassification
        # transform 정의 
        transform = ImageClassification(crop_size=args.img_size, resize_size=args.img_size)
    
    # 정규화에 사용될 mean과 std 불러옴
    else :
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

# getDataLoader
def getDataLoader(args): 
    
    # 데이터를 불러올 때 적용시킬 transform 불러옴
    transform = getTransform(args)
    
    # cifar Dataset 
    if args.data == 'cifar':
        train_dataset = CIFAR10(root='./cifar', train=True, transform=transform, download=True)
        test_dataset = CIFAR10(root='./cifar', train=False, transform=transform, download=True)

    # Dog dataset
    else:
        # ImageFolder
        if args.dataset == 'imagefolder':
            from torchvision.datasets import ImageFolder
            train_dataset = ImageFolder(root='/home/dataset/dog_v1_TT/train', transform=transform)
            test_dataset = ImageFolder(root='/home/dataset/dog_v1_TT/test', transform=transform)

        # Custom Dataset1 (일반적)
        # train, test가 나뉘어져 있는 경우 
        elif args.dataset == 'cust1':
            from .dogdataset import DogDataset
            train_dataset = DogDataset(root='/home/dataset/dog_v1_TT/train', transform=transform)
            test_dataset = DogDataset(root='/home/dataset/dog_v1_TT/test', transform=transform)
        
        # Custom Dataset2
        # train, test가 나뉘어져 있지 않은 경우
        # 전체 데이터를 아우르는 dataset class
        elif args.dataset == 'cust2':
            from sklearn.model_selection import train_test_split
            tmp_dataset = DogDataset(root='/home/dataset/dog_v1', transform=transform)
            # list로 반환됨 
            train_dataset, test_dataset = train_test_split(tmp_dataset, train_size= 0.8, random_state=2023)

    # dataloader 
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    # dataLoader 반환
    return train_loader, test_loader

# model_type에 따라 실행될 모델을 반환하는 함수 
# getTargetModel
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
        # 이 곳에서 pretrain 모델 불러옴
        ## if FineTuning
        if args.fine_tuning :
            from torchvision.models import resnet18
            from torchvision.models import ResNet18_Weights
            import torch.nn as nn
            # 모델과 weight 가져옴 
            weight = ResNet18_Weights
            model = resnet18(weight, progress=True) #  그냥 여기서 끝나면 아예 그대로 쓰는것일 뿐

            # Class 객체 접근 방법으로 수정하면 됨 
            # 모델의 최종 출력단 변경 
            ## 1000 이었던 class 개수를 현재 데이터에 맞게 바꿈 
            model.fc = nn.Linear(512, 5)
            model = model.to(args.device)
       
        ## 그렇지 않으면 
        else : 
            from networks.ResNet import ResNet
            model = ResNet(args).to(args.device) 

    # 이외에
    else : 
        # 일치하는 모델이 없을 경우 ValueError 띄움
        raise ValueError('no model implemented~')
    
    return model 