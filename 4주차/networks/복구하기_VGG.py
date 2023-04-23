import torch.nn as nn

# 5개의 유사한 구조의 VGGnet을 비교하기 위한 convolution 함수
class VGG_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size ):
        super().__init__()
        # VGGnet은 maxpool외에는 이미지의 크기 변화가 없음
        # 따라서 기본적으로 padding=1을 주고, kernel=1일 경우엔 padding이 없어도 됨
        padding = 1 if kernel_size==3 else 0
        # conv 구성
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=1, padding=padding)
        # BN
        self.bn = nn.BatchNorm2d(out_channels)
        # 활성함수
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
# 블록마다 반복되는 conv를 작성하는 함수 
class VGG_Block(nn.Moduel):
    def __init__(self, in_channels, output_channels, num_convs, last_1conv = False):
        super().__init__()
        # 공통으로 작성되는 첫번째 conv
        self.first_conv = VGG_conv(in_channels, output_channels)
        # 1conv=True일 때는 num_convs-2의 갯수만큼 middle 작성
        self.middel_convs = nn.ModuleList([
            VGG_conv(output_channels, output_channels) for _ in range(num_convs-2)
        ])
        # last_1conv에 따라 kenel size 수정 
        kernel_size = 1 if last_1conv else 3
        # 블록의 마지막 conv 작성
        self.last_convs = VGG_conv(output_channels, output_channels, kernel_size=kernel_size)
        # maxpooling
        self.mp = nn.MaxPool2d()

    def forward(self, x):
        x = self.first_conv(x)
        x = self.middel_convs(x)
        x = self.last_convs(x)
        x = self.mp(x)
        return x 

# classifier 작성
class VGG_classifier(nn.Module):
    def __init__(self, num_classes):
        super.__init__()
        # 논문에 맞게 Linear 작성  # 3번 적용
        self.fc = nn.Sequential(
            nn.Linear(32768, 4096),
            nn.Linear(4096, 4096),
            nn.Linear(4096, num_classes)
        )        

    def forward(self, x):
        x = self.fc(x)
        return x

# VGG_A는 블록당 conv가 1,1,2,2,2로 구성
class VGG_A(nn.Module):
    def __init__(self, num_classes):
        super.__init__()
        self.VGG_Block1 = VGG_Block(3, 64, 1)
        self.VGG_Block2 = VGG_Block(64, 128, 1)        
        self.VGG_Block3 = VGG_Block(128, 256, 2)        
        self.VGG_Block4 = VGG_Block(256, 512, 2)        
        self.VGG_Block5 = VGG_Block(512, 512, 2)
        self.FC = VGG_classifier(num_classes)      
        
    def forward(self, x):
        b, c, w, h = x.shape
        x = self.VGG_Block1(x)
        x = self.VGG_Block2(x)
        x = self.VGG_Block3(x)
        x = self.VGG_Block4(x)
        x = self.VGG_Block5(x)
        x = self.FC(x)

        return x 
    
# VGG_B는 VGG_A를 상속받음 (A의 forward를 사용->B이후부턴 def forward가 필요없음 )
# block1, block2가 2개씩으로 변경됨 
class VGG_B(VGG_A):
    def __init__(self, num_classes):
        super.__init__()
        self.VGG_Block1 = VGG_Block(3, 64, 2)
        self.VGG_Block2 = VGG_Block(64, 128, 2)

# VGG_C는 VGG_B를 상속받음
# block3, block4, block5의 conv가 3개씩으로 변경됨
# last1_conv = True 
class VGG_C(VGG_B):
    def __init__(self, num_classes):
        super.__init__()
        self.VGG_Block3 = VGG_Block(128, 256, 3, last_1conv=True)
        self.VGG_Block4 = VGG_Block(256, 512, 3, last_1conv=True)
        self.VGG_Block5 = VGG_Block(512, 512, 3, last_1conv=True)

# VGG_D는 VGG_B를 상속받음
# block3, block4, block5의 conv가 3개
class VGG_D(VGG_B):
    def __init__(self, num_classes):
        super.__init__()
        self.VGG_Block3 = VGG_Block(128, 256, 3)
        self.VGG_Block4 = VGG_Block(256, 512, 3)
        self.VGG_Block5 = VGG_Block(512, 512, 3)
 
# VGG_E는 VGG_D를 상속받음
# block3, block4, block5의 conv가 4개씩으로 변경됨 
class VGG_E(VGG_D):
    def __init__(self, num_classes):
        super.__init__()        
        self.VGG_Block3 = VGG_Block(128, 256, 4)
        self.VGG_Block4 = VGG_Block(256, 512, 4)
        self.VGG_Block5 = VGG_Block(512, 512, 4)