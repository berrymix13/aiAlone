import os 
import json
import torch 
import argparse

def parser_args(): 
    parser = argparse.ArgumentParser() 
    
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--hidden_size", type=int, default=500)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument("--img_size", type=int, default=32)

    parser.add_argument("--model_type", type=str, default='resnet', choices=['mlp', 'lenet', 'linear', 'conv', 'incep', 'vgg', 'resnet'])
    parser.add_argument("--vgg_type", type=str, default='a', choices=['a', 'b', 'c', 'd', 'e'])
    parser.add_argument("--res_config", type=str, default='18', choices=['18', '34', '50', '101', '152'])
    parser.add_argument("--fine_tuning", action='store_false')  #기본이 True
    parser.add_argument("--save_folder", type=str, default='results')
    parser.add_argument("--data", type=str, default='cifar', choices=['cifar', 'dog'])
    parser.add_argument("--dataset", type=str, default='imagefolder', choices=['imagefolder', 'cust1', 'cust2'])
    parser.add_argument("--save_iter", type=int, default=100)
    
    return parser.parse_args()

def infer_parser_args(): 
    parser = argparse.ArgumentParser() 
    
    parser.add_argument("--folder", type=str)
    parser.add_argument("--image", type=str)
    parser.add_argument("--device", default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    return parser.parse_args()

def load_trained_args(args): 
    with open(os.path.join(args.folder, 'args.json'), 'r') as f :
        trained_args = json.load(f)
    trained_args['device'] = args.device    
    trained_args = argparse.Namespace(**trained_args)
    return trained_args 