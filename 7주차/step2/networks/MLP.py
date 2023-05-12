# 필요 패키지 로드
import torch.nn as nn 

class myMLP(nn.Module): 
    def __init__(self, hidden_size, num_classes): 
        # 부모 class 명시 
        super().__init__()
        # fully connected layer : 이미지 분류기
        self.fc1 = nn.Linear(28*28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        b, c, w, h = x.shape  # 100, 1, 28, 28  
        # flatten 진행
        x = x.reshape(-1, 28*28) # 100, 28x28 
        # x = x.reshape(b, -1)
        # hidden layer에 x 전달
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x
