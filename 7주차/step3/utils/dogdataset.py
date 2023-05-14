import torch, os
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image

# CustomDataset 1 
# train, test가 나뉘어져 있는 경우 
class DogDataset(Dataset):
    def __init__(self, root, transform) :
        super().__init__()
        # class의 정보(name)
        # 숨김파일이 포함되지 않도록 불러오기
        self._classes = [c for c in os.listdir(root) if '.DS_Store' not in c]
        # class와 index 정보 매치 (dict)
        self.class2idx = {c:i for i, c in enumerate(self._classes)}
        self.idx2class = {i:c for i, c in enumerate(self._classes)}
        # 객체 변수로 tranform 변수 선언 
        self.transform = transform
        # 전체 데이터를 하나의 리스트(A)로 넣어주기
        self.image_pathes = []
        # class name 을 활용한 for문
        for name in self._classes:
            # _class_path 정의
            _class_path = os.path.join(root, name)
            # _class_path 내의 파일명을 불러오는 for문
            for img_name in os.listdir(_class_path):
                # _class_path와 img_name을 결합해 img_path 생성
                image_path = os.path.join(_class_path, img_name)
                # image_pathes에 image_path 추가 
                self.image_pathes.append(image_path)

        
    ## --- len --- ##
    def __len__(self):
        # A의 길이를 return
        return len(self.image_pathes)

    ## -- getitem -- ##
    def __getitem__(self, idx):
        super().__init__()
        # index를 기반으로 list
        image_path = self.image_pathes[idx]
        # 리스트 A에서 index를 indexing해서 가져오면 이미지의 경로가 나옴
        # 이미지 경로를 바탕으로 PIL Image 객체를 생성 
        # png(4채널), jpg(3채널) #png를 수정해줘야 됨
        image = Image.open(image_path).convert('RGB')
        
        # 사이즈와 같은 transformation (transform:위에서) 하나하나 불러옴
        image = self.transform(image)
        # 이 이미지의 정답이 뭔지 뽑아야 함 
        _class = os.path.basename(os.path.dirname(image_path))

        # 정답의 index로 변환 
        target = self.class2idx[_class]

        ## 중요
        return image, target 