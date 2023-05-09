import torch, os
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image

# CustomDataset 1 
class DogDataset(Dataset):
    def __init__(self, root, transform) :
        super().__init__()
        self._classes = [c for c in os.listdir(root) if '.DS_Store' not in c]
        self.class2idx = {c:i for i, c in enumerate(self._classes)}
        self.idx2class = {i:c for i, c in enumerate(self._classes)}
        self.transform = transform
        self.image_pathes = []
        for name in self._classes:
            _class_path = os.path.join(root, name)
            for img_name in os.listdir(_class_path):
                image_path = os.path.join(_class_path, img_name)
                self.image_pathes.append(image_path)

        
    def __len__(self):
        return len(self.image_pathes)

    def __getitem__(self, idx):
        super().__init__()
        image_path = self.image_pathes[idx]
        image = Image.open(image_path).convert('RGB')
        
        image = self.transform(image)
        _class = os.path.basename(os.path.dirname(image_path))

        target = self.class2idx[_class]

        return image, target 