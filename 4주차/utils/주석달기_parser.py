import json, os
import torch
import argparse

# parser 함수 정의 
def parser_args():
    parser = argparse.ArgumentParser()
    # batch_size
    parser.add_argument("--batch_size", type=int, default=100)
    # hidden_size
    parser.add_argument("--hidden_size", type=int, default=500)
    # num_classes
    parser.add_argument("--num_classes", type=int, default=10)
    # lr
    parser.add_argument("--lr", type=float, default=1e-3)
    # epochs
    parser.add_argument("--epochs", type=int, default=3)
    # device
    parser.add_argument("--device", default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    # img_size
    parser.add_argument("--img_size", type=int, default=32)
    # model_type
    parser.add_argument('--model_type', type=str, default='mylenet', choices=['mlp', 'lenet', 'linear', 'conv', 'incep'])
    # parser.add_argument("--vgg_type", type=str, default='a', choices=['a', 'b', 'c', 'd', 'e'])
    # save_folder (results라는 폴더 생성)
    parser.add_argument("--save_folder", type=str, default="results")
    
    return parser.parse_args()

# 추론에 사용될 parser 선언
def infer_parser_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--folder", type=str)
    parser.add_argument("--image", type=str)
    parser.add_argument("--device", default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    return parser.infer_parse_args()

# 저장된 best_model의 json file을 읽어오는 함수
def load_trained_args(args):
    # json 불러옴 
    with open(os.path.join(args.folder,'args.json'), 'r') as f:
        trained_args = json.load(f)
    # device 정의 
    trained_args['device'] = args.device
    # ** : 딕셔너리 형태로 입력이 가능하게 해줌
    trained_args = argparse.Namespace(**trained_args)
    return trained_args