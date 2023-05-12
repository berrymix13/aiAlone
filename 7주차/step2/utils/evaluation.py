# 필요 패키지 로드 
import torch  

# eval
def eval(model, loader, args):
    # total, correct 변수 선언
    total = 0 
    correct = 0
    for _, (image, target) in enumerate(loader):

        image = image.to(args.device)
        target = target.to(args.device)

        # 결과 저장 
        out = model(image)
        # 가장 높은 확률의 index를 구함
        _, pred = torch.max(out, 1)
        # 예측값과 target이 일치하는 것들만 합함.  
        correct += (pred == target).sum().item()
        # 각 배치에서 배치 사이즈 = 각 배치의 전체 데이터 수 
        total += image.shape[0]
    
# 정확도(acc)  : (전체 정답 건수) / (전체 데이터 수)
    return correct / total 

# 각 클래스 마다의 정확도 계산
def eval_class(model, loader, args):
    # 각각의 class 마다의 정확도를 계산하기 휘해 num_class 갯수만큼 torch생성
    total = torch.zeros(args.num_classes) 
    correct = torch.zeros(args.num_classes) 

    for idx, (image, target) in enumerate(loader):
        image = image.to(args.device)
        target = target.to(args.device)
        
        out = model(image)
        _, pred = torch.max(out, 1)
        # 각 클래스 마다 값을 입력하기 위한 for문   
        for i in range(args.num_classes): 
            # target이 i와 같고, 예측 또한 i와 같을 때 더함.
            correct[i] += ((target == i) & (pred == i)).sum().item()
            # 각 class가 i와 같을 때만 더함 
            total[i] += (target == i).sum().item()
    return correct, total 