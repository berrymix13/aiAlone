import os, sys
# 현재 위치를 추가해줌 (상위폴더 import 에러시 사용)
sys.path.append(os.getcwd())

import json
import torch
import torch.nn as nn
from torch.optim import Adam

from utils.주석달기_parser import parser_args
from utils.주석달기_getModeuls import getDataLoader, getTargetModel
from utils.evaluation import eval
from utils.주석달기_tools import get_save_folder_path

def main():
    # arge 정의
    args = parser_args() 
    # 모델을 저장할 폴더 경로 정의
    save_folder_path = get_save_folder_path(args)

    # 결과 폴더 생성
    os.makedirs(save_folder_path)
    # json파일 생성 
    with open(os.path.join(save_folder_path, 'args.json'), 'w') as f:
        # json_args에 args 복사
        json_args = args.__dict__.copy()
        # json 저장에 오류나는 device 삭제
        del json_args['device']
        # dump   : python 객체를 Json문자열로 변환
        # indent : 들여쓰기 갯수 (미관용)
        json.dump(json_args, f, indent=4)  
    
    # train_loader, test_loader 정의
    train_loader, test_loader = getDataLoader(args) 

    # model 정의
    model =getTargetModel(args)

    # Resume 작성해보기 (수정)
    # 참고자료에서는 DataLoad, 모델정의 이후, optim정의 이전에 작성됨
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('result'), 'Error: no result directory found!'
        # result폴더 내 가장 큰 번호 찾기
        current_max_value = max([int(f) for f in os.listdir(args.save_folder)])
        # 가장 큰 번호에서 checkpoint 가져옴
        checkpoint = torch.load(f'./result/{current_max_value}.pth')
        # 가중치 업데이트
        # 수정 : checkpoint['net]  -> checkpoint
        model.load_state_dict(checkpoint)
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    # loss 정의
    loss = nn.CrossEntropyLoss()
    # optimezer 정의
    optim = Adam(model.parameters(), lr=args.lr)

    # 정확도 저장용 변수 정의
    best_acc = 0
    for epoch in range(start_epoch, start_epoch+200):
        for idx, (image, target) in enumerate(train_loader):
            image = image.to(args.device)
            target = target.to(args.device)

            out = model(image) 
            loss_value = loss(out, target)

            optim.zero_grad()
            loss_value.backward()
            optim.step()

            if idx % 100 == 0:
                print(loss_value.item())
                acc = eval(model, test_loader, args)[0]
                print('acc : ',acc)
                # 현재 정확도가 best정확도보다 높으면 저장
                if best_acc < acc :
                    # best_acc 업데이트
                    best_acc = acc
                    # state_dict : 각 layer마다 tensor로 매핑되는 매개변수를 dict로 저장
                    torch.save(model.state_dict(), os.path.join(save_folder_path, f'best_model.ckpt'))
                    print(f'new best model saved!! acc:{acc*100:.2f}')

# main함수 실행 
if __name__ == '__main__':
     main()

