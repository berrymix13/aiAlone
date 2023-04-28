# 필요 패키지 로드
import os 
import json
import torch 
import argparse

# 인자값 저장을 위한 함수 
def parser_args(): 
    # 인자값을 받을 수 있는 인스턴스 생성
    parser = argparse.ArgumentParser() 
    
    ### --- 입력받을 인자값 등록 --- ##
    # batch_size
    parser.add_argument("--batch_size", type=int, default=100)
    # hidden_size
    parser.add_argument("--hidden_size", type=int, default=500)
    # num_classes
    parser.add_argument("--num_classes", type=int, default=10)
    # lr
    parser.add_argument("--lr", type=float, default=0.001)
    # epochs
    parser.add_argument("--epochs", type=int, default=3)
    # device
    parser.add_argument("--device", default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    # img_size
    parser.add_argument("--img_size", type=int, default=32)

    # model_type
    parser.add_argument("--model_type", type=str, default='lenet', choices=['mlp', 'lenet', 'linear', 'conv', 'incep', 'vgg', 'resnet'])
    # vgg_type
    parser.add_argument("--vgg_type", type=str, default='a', choices=['a', 'b', 'c', 'd', 'e'])
    # res_config
    parser.add_argument("--res_config", type=str, default='18', choices=['18', '34', '50', '101', '152'])
    # save_folder
    parser.add_argument("--save_folder", type=str, default='results')
    
    # return
    return parser.parse_args()

# 추론에 사용될 parser 선언
def infer_parser_args(): 
    # 인자값을 받을 수 있는 인스턴스 생성
    parser = argparse.ArgumentParser() 
    
    ### --- 입력받을 인자값 등록 --- ##
    # folder
    parser.add_argument("--folder", type=str)
    # image
    parser.add_argument("--image", type=str)
    # device
    parser.add_argument("--device", default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # return
    return parser.parse_args()

# 저장된 best_model의 json file을 읽어오는 함수
def load_trained_args(args): 
    # json 불러옴 
    with open(os.path.join(args.folder, 'args.json'), 'r') as f :
        trained_args = json.load(f)
    # device 정의 
    trained_args['device'] = args.device    
    # ** : dictionary 값을 un-packing
    trained_args = argparse.Namespace(**trained_args)
    return trained_args 