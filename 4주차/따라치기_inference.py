import os, torch
from PIL import Image
import torch.nn.functional as F
from utils.parser import infer_parser_args, load_trained_args
from utils.getModeuls import getTransform, getTargetModel

def main():
    args = infer_parser_args()
    
    assert os.path.exists(args.folder), "학습 폴더 넣어라"
    assert os.path.exists(args.images), "추론할 이미지 넣어라"
    
    trained_args = load_trained_args(args)
    model = getTargetModel(trained_args)
    model.load_state_dict(torch.load(os.path.join(args.folder, "best_model.ckpt")))

    transform = getTransform(trained_args)

    img = Image.open(args.image)
    img = transform(img)
    img = img.unsqueeze(0)

    output = model(img)

    prob = F.softmax(output, dim=1)
    index = torch.argmax(prob) # index
    value = torch.max(prob) # 값
    class_names = ['airplanes', 'cars', 'birds', 'cats', 'deer', 
                   'dogs', 'frogs', 'horses', 'ships', 'trucks']
    print(f'Image is {class_names[index]}, and to confidence is {value*100:.2f}%')

if __name__ == "__main__":
    main()