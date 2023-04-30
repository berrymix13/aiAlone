## --- 패키지 불러오기 --- ## 
import sys, os
# 현재 경로 추가
sys.path.append(os.getcwd())
import json 
import torch 
import torch.nn as nn 
from torch.optim import Adam 

from utils.parser import parser_args 
from utils.getModules import getDataLoader 
from utils.getModules import getTargetModel 
from utils.evaluation import eval 
from utils.tools import get_save_folder_path

# 메인 함수 정의
def main():
    # args 정의
    args = parser_args() 

    ## --- 학습세팅 hparams 저장 --- ##
    # 저장 경로 설정
    save_folder_path = get_save_folder_path(args)
    # 저장경로에 파일 생성(존재하지 않으면 지정 폴더를 생성함)
    os.makedirs(save_folder_path)
    # args.json파일 작성
    with open(os.path.join(save_folder_path, 'args.json'), 'w') as f : 
        # json_args에 args의 속성정보를 copy함
        json_args = args.__dict__.copy()
        # copy된 json_args에서 device항목 삭제(오류해결용)
        del json_args['device']
        # 파이썬 객체 직렬화 (보기좋게 만듦)
        json.dump(json_args, f, indent=4)

    # Data loader 불러옴ㄴ
    train_loader, test_loader = getDataLoader(args) 

    # 모델, loss, optimizer 정의
    model = getTargetModel(args) 
    loss = nn.CrossEntropyLoss() 
    optim = Adam(model.parameters(), lr=args.lr) 

    ## --- 학습 loop 생성 --- ##
    # len(dataset) % batch_size = 1 epoch
    # 최고 정확도를 저장할 변수 생성
    best_acc = 0
    for epoch in range(args.epochs): 
        for idx, (image, target) in enumerate(train_loader): 
            # 데이터에 학습을 처리할 디바이스 지정
            image = image.to(args.device) 
            target = target.to(args.device) 

            # 학습 결과 저장
            out = model(image) 
            loss_value = loss(out, target) 

            ## 역전파 
            # 가중치 업데이트에 앞서 계산된 미분값의 누적을 막음
            optim.zero_grad() 
            # tensor에 대한 자동 미분 : 기울기 값 알아냄
            loss_value.backward() 
            # parameter 업데이트 
            optim.step() 

            # batch가 100번 돌 때마다 loss 표시 
            if idx % 100 == 0 : 
                print(loss_value.item())
                acc = eval(model, test_loader, args)
                print('accuracy : ', acc)

                # 현재 모델의 정확도가 best_acc보다 높다면 
                if best_acc < acc :
                    # best_acc 갱신
                    best_acc = acc 
                    # 모델을 새로 저장함
                    torch.save(model.state_dict(), os.path.join(save_folder_path, f'best_model.ckpt'))
                    # 저장시 문구와 정확도  출력
                    print(f'new best model saved! acc : {acc*100:.2f}')

# main 함수 실행
if __name__ == '__main__': 
    main()
