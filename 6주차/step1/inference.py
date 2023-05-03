# 패키지 불러오기 
import os 
import torch 

from PIL import Image
import torch.nn.functional as F

from utils.parser import infer_parser_args 
from utils.parser import load_trained_args 
from utils.getModules import getTargetModel 
from utils.getModules import getTransform

def main():
    pass 
    # 추론 전용 파서 불러오기 
    args = infer_parser_args() 

    assert os.path.exists(args.folder), "학습 폴더 넣어라"
    assert os.path.exists(args.image), "추론할 이미지도 넣어라"

    # 학습이 된 폴더를 기반으로 학습된 args 불러오기 
    trained_args = load_trained_args(args)

    # 모델을 학습된 상황에 맞게 재설정 
    model = getTargetModel(trained_args) 

    # 모델 weight 업데이트 
    model.load_state_dict(torch.load(os.path.join(args.folder, 'best_model.ckpt')))

    # 데이터 전처리 코드 준비 
    transform = getTransform(trained_args) 

    # 이미지 불러오기 
    img = Image.open(args.image)
    img = transform(img)
    img = img.unsqueeze(0)

    # 모델 출력 
    output = model(img)

    # 결과 후처리 
    prob = F.softmax(output, dim=1)
    index = torch.argmax(prob)
    value = torch.max(prob)

    classes = ['airplanes', 'cars', 'birds', 'cats', 'deer', 'dogs', 'frogs', 'horses', 'ships', 'trucks']
    print(f'Image is {classes[index]}, and the confidence is {value*100:.2f} %')

if __name__ == '__main__': 
    main()
