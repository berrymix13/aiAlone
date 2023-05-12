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
    ### 그냥 가져와도 됨
    args = parser_args()    ## Fine Tuning을 함

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

    # Data loader 불러옴
    train_loader, test_loader = getDataLoader(args) 
    ### 여기까지 그대로 

    # 모델, loss 정의
    model = getTargetModel(args)    # pytorch에서 구현한 ResNet18 + 뒤 쪽의 MLP변경
    loss = nn.CrossEntropyLoss() 
    # param_groups (list)
    param_groups = [
        # dict 형태로 원소 넣어줌
        {'params':module, 'lr':args.lr * 0.01}
            # named_parameters  : 모델의 이름과 모듈 출력
            for name, module in model.named_parameters() 
            # fc가 name에 포함되지 않았다면 
            if 'fc' not in name 
    ]
    # param_groups에 새로 추가 
    param_groups.append({'params':model.fc.parameters(), 'lr':args.lr})
    # optim 정의 
    optim = Adam(param_groups, lr = args.lr)
    # model.parameters() 처럼 한번에 넣지 않고 list 형태로 넣음 
    # param 그룹의 형태로 넣어주어야 됨 
    # resnet의 기존 파라미터는 lr=1e-5, 뒤에 붙은 MLP는 큰 lr

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
            if idx % args.save_iter == 0 : 
            # if idx % (len(dataloader) // 3) == 0 :
                print(loss_value.item())
                acc = eval(model, test_loader, args)
                print('accuracy : ', acc)

                # 현재 모델의 정확도가 best_acc보다 높다면 
                if best_acc < acc :
                    # best_acc 갱신
                    best_acc = acc 
                    # 모델을 새로 저장함
                    torch.save(model.state_dict(), 
                               os.path.join(save_folder_path, f'best_model.ckpt'))
                    # 저장시 문구와 정확도  출력
                    print(f'new best model saved! acc : {acc*100:.2f}')

# main 함수 실행
if __name__ == '__main__': 
    main()
