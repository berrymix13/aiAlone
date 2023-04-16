import torch.nn as nn
from torch.optim import Adam

from utils.복구하기_parser import parser_args
from utils.복구하기_getModuels import getDataLoader, getTargetModel
from utils.복구하기_evalution import eval

# 상위 폴더 import 에러시 사용
import os, sys
# list 형태인 sys.path에 현재 경로 추가
sys.path.append(os.getcwd())

# 주요 기능이 실행되는 main함수 
    # return받은 인자값들을 args에 저장 
def main():
    args = parser_args()
    train_loader, test_loader = getDataLoader(args)

    model = getTargetModel(args)
    loss = nn.CrossEntropyLoss()
    optim = Adam(model.parameters(), lr = args.lr)

    for epoch in range(args.epochs):
        for idx, (image, target) in enumerate(train_loader):
            image = image.to(args.device)
            target = target.to(args.device)

            out = model(image)
            loss_value = loss(out, target)
            
            optim.zero_grad()
            loss_value.backward()
            optim.step()

            if idx % 100 == 0:
                print(f'loss_value  : {loss_value.item()}')
                print(f'accuracy    : {eval(model, test_loader, args)}')

# 메인함수의 선언 (시작)
if __name__ == '__main__':
    main()