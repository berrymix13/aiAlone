import torch 
import torch.nn as nn

# 기본 Lenet
class mylenet(nn.Module):
     def __init__(self, num_classes): 
          super().__init__()
          # 컬러이미지에 맞게 in_channels 설정  
          self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5) 
          # out_channels 갯수만큼 채널 각각에서 평균을 낸다
          # 학습과정에서 각 배치단위 별로 데이터의 다양한 분포를 정규화 시킴 
          self.bn1 = nn.BatchNorm2d(num_features=6)
          # 입력 값을 다음 노드에 전달하는 활성 함수(activated func.)
          self.act1 = nn.ReLU() 
          
          # 이미지 사이즈 절반으로 줄임.
          self.pool1 = nn.MaxPool2d(kernel_size=2) 
          
          # Conv2 
          self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5) 
          # bn2
          self.bn2 = nn.BatchNorm2d(num_features=16)
          # activ2
          self.act2 = nn.ReLU()
          # pooling2
          self.pool2 = nn.MaxPool2d(kernel_size=2) 

          # 이미지 분류기1
          self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)  
          # 이미지 분류기2
          self.fc2 = nn.Linear(in_features=120, out_features=84)
          # 이미지 분류기3
          self.fc3 = nn.Linear(in_features=84, out_features=num_classes)
    
     def forward(self, x):
          b, c, h, w = x.shape 
          x = self.conv1(x)
          x = self.bn1(x)
          x = self.act1(x)
          x = self.pool1(x)

          x = self.conv2(x)
          x = self.bn2(x)
          x = self.act2(x)
          x = self.pool2(x)

          x = x.reshape(b, -1)
          x =self.fc1(x)
          x =self.fc2(x)
          x =self.fc3(x)

          return x

# 신경망 모듈의 그룹화 1 : Sequential
class myLeNet_seq(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Sequential : 모듈의 순서를 고정함
        self.seq1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5) ,
            nn.BatchNorm2d(num_features=6),
            nn.ReLU() ,

            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5) ,
            nn.BatchNorm2d(num_features=16),
            nn.ReLU() ,

            nn.MaxPool2d(kernel_size=2),
        )
        
        # forward 내에서 reshape이 일어나기 전과 후로 나누어 묶음
        self.seq2 = nn.Sequential(
            nn.Linear(in_features=16*5*5, out_features=120),
            nn.Linear(in_features=120, out_features=84),
            nn.Linear(in_features=84, out_features=num_classes),
        ) 
        
    def forward(self, x): 
        b, c, h, w = x.shape
        x = self.seq1(x)
        x = x.reshape(b, -1)
        x = self.seq2(x)
        return x 


# Conv 사이에 Linear 삽입
class mylenet_Linear(nn.Module):
     def __init__(self, num_classes):
          super().__init__()
          self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5) 
          self.bn1 = nn.BatchNorm2d(num_features=6)  
          self.act1 = nn.ReLU()
          self.pool1 = nn.MaxPool2d(kernel_size=2) 
          
          # 전체적인 맥락 판단을 위해 중간에 Linear를 삽입, 이후 동일
          self.fc_1 = nn.Linear(6*14*14, 2048)
          self.fc_2 = nn.Linear(2048, 6*14*14)
     
          self.conv2 =  nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
          self.bn2 = nn.BatchNorm2d(num_features=16)  
          self.act2 = nn.ReLU()
          self.pool2 = nn.MaxPool2d(kernel_size=2) 

          self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)  
          self.fc2 = nn.Linear(in_features=120, out_features=84)
          self.fc3 = nn.Linear(in_features=84, out_features=num_classes)
    
     def forward(self, x):
          b, c, h, w = x.shape 

          x = self.conv1(x)
          x = self.bn1(x)
          x= self.act1(x)
          x = self.pool1(x)

          # Linear 진행을 위해 shape 변경이 필요함.
          _, tmp_c, tmp_w, tmp_h = x.shape
          x = x.reshape(b, -1) 
          x = self.fc_1(x)
          x = self.fc_2(x)
          # 다시 원래 shape으로 돌려 계속 진행함 
          x = x.reshape(b, tmp_c, tmp_w, tmp_h) 

          x = self.conv2(x)
          x = self.bn2(x)
          x= self.act2(x)
          x = self.pool2(x)
          
          # flatten 진행
          x = x.reshape(b, -1)
          # 이미지 분류기에 입력 
          x =self.fc1(x)
          x =self.fc2(x)
          x =self.fc3(x)

          return x

class mylenet_convs(nn.Module):
    def __init__(self, num_classes): 
        super().__init__()
        # 신경망 모듈의 그룹화 2 : ModuleList
        # 이미지의 사이즈를 유지한 채로 Conv 를 세 번 반복한다. 
        self.tmp_conv1 = nn.ModuleList(
            # in_channel의 수가 2번째 부터 변경되기 때문에 1번째를 따로 작성하여 진행했다. 
            [nn.Conv2d(3, 6, 3, 1, 1)] + [
            nn.Conv2d(6, 6, 3, 1, 1) for _ in range(3-1)
        ])
        # 이하 동일 
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=5) 
        self.bn1 = nn.BatchNorm2d(num_features=6)
        self.act1 = nn.ReLU() 

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5) 
        self.bn2 = nn.BatchNorm2d(num_features=16)
        self.act2 = nn.ReLU() 

        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x): 
        b, c, h, w = x.shape
        # ModuleList또한 List이기 때문에, List를 사용하듯 for문으로 접근한다. 
        for module in self.tmp_conv1: 
            x = module(x)

        # 이하 동일 
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x) 
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x) 
        x = self.pool2(x)

        x = x.reshape(b, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x 

# Conv를 병렬적으로 사용한 뒤 통합하는 방법 
class mylenet_incep(nn.Module):
    def __init__(self, num_classes): 
        super().__init__()

        # k를 각각 5, 3, 1로 설정
        self.conv1_1 = nn.Conv2d(3, 6, 5, 1, 2)
        self.conv1_2 = nn.Conv2d(3, 6, 3, 1, 1)
        self.conv1_3 = nn.Conv2d(3, 6, 1, 1, 0)

        # 각각의 ch수가 6이므로 in_ch은 6*3이 됨.
        self.conv1 = nn.Conv2d(in_channels=18, out_channels=6, kernel_size=5) 
        # 이하 동일 
        self.bn1 = nn.BatchNorm2d(num_features=6)
        self.act1 = nn.ReLU() 

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5) 
        self.bn2 = nn.BatchNorm2d(num_features=16)
        self.act2 = nn.ReLU() 

        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x): 
        b, c, h, w = x.shape
        x_1 = self.conv1_1(x)
        x_2 = self.conv1_2(x)
        x_3 = self.conv1_3(x)

        # 세 개의 conv를 병합함. 6+6+6이 되도록 합칠 때의 dimension 설정 필요 
        x_cat = torch.cat((x_1, x_2, x_3), dim=1)
        x = self.conv1(x_cat)
        x = self.bn1(x)
        x = self.act1(x) 
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x) 
        x = self.pool2(x)

        x = x.reshape(b, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
 