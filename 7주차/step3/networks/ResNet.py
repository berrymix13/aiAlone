import torch 
import torch.nn as nn 

# 18 - > 34 시 변하는 부분
NUM_BLOCKS_18 = [2, 2, 2, 2]
NUM_BLOCKS_34 = [3, 4, 6, 3]
NUM_BLOCKS_50 = [3, 4, 6, 3]
NUM_BLOCKS_101 = [3, 4, 23, 3]
NUM_BLOCKS_152 = [3, 8, 36, 3]
NUM_CHANNEL_33 = [64, 64, 128, 256, 512]
NUM_CHANNEL_131 = [64, 256, 512, 1024, 2048]


# batch가 작을 때는 BN을 쓰면 안됨!
# 다른 방법으로 normalize를 해야 됨
# 적어도 두 자릿수는 되어야 BN을 사용함 
 
# 입력부 # 공통으로 나타남
class ResNet_front(nn.Module):
    def __init__(self): 
        super().__init__()
        # conv
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # pool
        self.pool = nn.MaxPool2d(3, 2, 1)
    
    # forward
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x 
    
# 출력부 # 공통으로 나타남
class ResNet_back(nn.Module):
    # config   : (str)
    def __init__(self, num_classes=10, config='18'): 
        super().__init__()
        ## pool 
        # AdaptiveAvgPool2d : 어떤 사이즈로 출력할지 미리 정할 수 있음
        self.pool = nn.AdaptiveAvgPool2d(1)
        # config에 따라 fc의 입력이 달라짐
        in_feat = 512 if config in ['18', '34'] else 2048
        # fc
        self.fc = nn.Linear(in_feat, num_classes)
    
    # forward
    def forward(self, x):
        x = self.pool(x) 
        # 차원 줄임
        # 맨 마지막 자리를 512로 맞추기 위함
        x = torch.squeeze(x)
        x = self.fc(x) 
        return x 
    
# 3-3 구조
class ResNet_Block(nn.Module):
    def __init__(self, in_channel, out_channel, downsampling=False): 
        super().__init__()
        # downsampling 정의
        self.downsampling = downsampling 
        # stride 정의
        stride = 1
        # downsampling = True 일 때:
        if self.downsampling : 
            # stride 재정의
            stride = 2
            # skip_conv 정의
            self.skip_conv = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 3, stride, 1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU()
            )

        # first_conv
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

        # second_conv
        self.second_conv = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

        # return
        self.relu = nn.ReLU()

    # forward
    def forward(self, x):
        # clone   : 기존 tensor와 내용을 복사한 tensor를 생성함
        skip_x = torch.clone(x) 
        # downsampling = True일 때 :
        if self.downsampling : 
            skip_x = self.skip_conv(skip_x)

        x = self.first_conv(x)
        x = self.second_conv(x)

        # 기존 x와 skip_x를 더함 
        x = x + skip_x
        x = self.relu(x)

        return x 
    

# 1-3-1 구조 
class ResNet_BottleNeck(nn.Module): 
    def __init__(self, in_channel, out_channel, downsampling=False): 
        super().__init__()
        # downsamplimg, stride 정의
        self.downsampling = downsampling
        stride = 2 if downsampling else 1 

        # skip_conv # 3-3구조와 동일
        self.skip_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

        # third_conv과 first, second conv의 out_channel은 4배 차이남
        # first_conv
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel // 4, 1, 1, 0),
            nn.BatchNorm2d(out_channel // 4),
            nn.ReLU()
        )

        # 크기 변화가 가능한 부분 (1x1은 크기 변화가 어려움)
        self.second_conv = nn.Sequential(
            nn.Conv2d(out_channel // 4, out_channel // 4, 3, stride, 1),
            nn.BatchNorm2d(out_channel // 4),
            nn.ReLU()
        )

        # third_conv
        self.third_conv = nn.Sequential(
            nn.Conv2d(out_channel // 4, out_channel, 1, 1, 0),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

        # relu
        self.relu = nn.ReLU()

    # forward
    def forward(self, x):

        # x 복제해서 skip_x 정의
        skip_x = torch.clone(x) 
        skip_x = self.skip_conv(skip_x)

        x = self.first_conv(x)
        x = self.second_conv(x)
        x = self.third_conv(x)

        x = x + skip_x
        x = self.relu(x)

        return x 
    

# ResNet_middele
class ResNet_middle(nn.Module):
    def __init__(self, config): 
        super().__init__()
        # 입력 숫자, 출력 숫자, block 갯수 
        if config == '18': 
            num_blocks, num_channel  = NUM_BLOCKS_18, NUM_CHANNEL_33
            self.target_layer = ResNet_Block
        elif config == '34': 
            num_blocks, num_channel  = NUM_BLOCKS_34, NUM_CHANNEL_33
            self.target_layer = ResNet_Block
        elif config == '50': 
            num_blocks, num_channel  = NUM_BLOCKS_50, NUM_CHANNEL_131
            self.target_layer = ResNet_BottleNeck
        elif config == '101': 
            num_blocks, num_channel  = NUM_BLOCKS_101, NUM_CHANNEL_131
            self.target_layer = ResNet_BottleNeck
        elif config == '152': 
            num_blocks, num_channel  = NUM_BLOCKS_152, NUM_CHANNEL_131
            self.target_layer = ResNet_BottleNeck

        # 입력 숫자, 출력 숫자, block 갯수 
        # downsampling - True : 공간적인 정보를 반으로 줄임
        self.layer1 = self.make_layer(num_channel[0], num_channel[1], num_blocks[0])
        self.layer2 = self.make_layer(num_channel[1], num_channel[2], num_blocks[1], True)
        self.layer3 = self.make_layer(num_channel[2], num_channel[3], num_blocks[2], True)
        self.layer4 = self.make_layer(num_channel[3], num_channel[4], num_blocks[3], True)
    
    # make_layer
    def make_layer(self, in_channel, out_channel, num_block, downsampling=False): 
        # downsampling은 처음 layer에만 들어감
        layer = [ self.target_layer(in_channel, out_channel, downsampling) ]
        # 나머지 layer 생성
        for _ in range(num_block - 1):
            layer.append(self.target_layer(out_channel, out_channel))
        return nn.Sequential(*layer) 
    
    # forward
    def forward(self, x):
        x = self.layer1(x) 
        x = self.layer2(x) 
        x = self.layer3(x) 
        x = self.layer4(x) 
        return x 

# ResNet 취합 
class ResNet(nn.Module):
    def __init__(self, args): 
        super().__init__() 
        self.front = ResNet_front()
        self.middle = ResNet_middle(args.res_config)
        self.back = ResNet_back(args.num_classes, args.res_config)
    
    def forward(self, x):
        x = self.front(x)
        x = self.middle(x)
        x = self.back(x)
        return x 
    

