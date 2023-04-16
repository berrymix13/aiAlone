# argparse  : 인자 파싱에 사용되는 모듈 
import argparse, torch 

# 인자값 저장을 위한 함수 
def parser_args():
    # 인자값을 받을 수 있는 인스턴스 생성
    parser = argparse.ArgumentParser()

    # 입력받을 인자값 등록 
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--hidden_size", type=int, default=500)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument("--img_size", type=int, default=32)
    parser.add_argument("--model_type", type=str, default='lenet',choices=['mlp', 'lenet', 'linear', 'conv', 'incep'])
    
    # 인자값 return
    return parser.parse_args()  