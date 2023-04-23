import os, torch
from PIL import Image
import torch.nn.functional as F
from utils.복구하기_parser import infer_parser_args, load_trained_args
from utils.복구하기_getModules import getTransform, getTargetModel

def main():
    # 추론 저전용 파서 정의
    args = infer_parser_args()
    
    # assert : 가정 설정문
    # folder 예외 상황 처리
    assert os.path.exists(args.folder), "학습 폴더 넣어라"
    # image 예외 상황 처리
    assert os.path.exists(args.images), "추론할 이미지 넣어라"
    
    # 학습된 args 불러옴
    trained_args = load_trained_args(args)
    
    # 모댈을 학습된 상황에 맞게 재설정 
    model = getTargetModel(trained_args)
    
    # 모델 가중치 업데이트 
    model.load_state_dict(torch.load(os.path.join(args.folder, "best_model.ckpt")))

    # 데이터 전처리 코드 준비
    transform = getTransform(trained_args)

    # 이미지 불러오기 
    img = Image.open(args.image)
    img = transform(img)
    img = img.unsqueeze(0)

    # 모델 출력 
    output = model(img)

    # 결과 후처리 
    # prob
    prob = F.softmax(output, dim=1)
    # index
    index = torch.argmax(prob) 
    # value
    value = torch.max(prob) 
    # classes
    class_names = ['airplanes', 'cars', 'birds', 'cats', 'deer', 
                   'dogs', 'frogs', 'horses', 'ships', 'trucks']
    # 추론 결과 출력
    print(f'Image is {class_names[index]}, and to confidence is {value*100:.2f}%')

if __name__ == "__main__":
    main()