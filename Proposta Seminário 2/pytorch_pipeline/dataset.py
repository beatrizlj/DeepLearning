import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from config import IMAGE_SIZE

def load_medmnist_data(npz_path):
    """
    Carrega o dataset MedMNIST do arquivo .npz.
    Retorna as imagens combinadas (train + val + test) e as labels
    para podermos aplicar o K-Fold Cross Validation em todo o dataset.
    """
    data = np.load(npz_path)
    
    # Combinar tudo para o K-Fold
    images = np.concatenate([data['train_images'], data['val_images'], data['test_images']], axis=0)
    labels = np.concatenate([data['train_labels'], data['val_labels'], data['test_labels']], axis=0)
    
    # As labels vêm como (N, 1), transformamos em (N,)
    labels = labels.squeeze()
    
    # Nomes das classes fixos do BloodMNIST
    idx_to_class = {
        0: 'basophil', 1: 'eosinophil', 2: 'erythroblast', 3: 'ig',
        4: 'lymphocyte', 5: 'monocyte', 6: 'neutrophil', 7: 'platelet'
    }
    
    return images, labels, idx_to_class


class BloodCellDataset(Dataset):
    def __init__(self, images, labels, is_train=True):
        self.images = images
        self.labels = labels
        self.is_train = is_train
        
        # Transforms padrão do ImageNet (necessários para Transfer Learning)
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        
        if self.is_train:
            self.transforms = T.Compose([
                T.Resize(IMAGE_SIZE),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(20),
                T.ToTensor(),
                normalize
            ])
        else:
            self.transforms = T.Compose([
                T.Resize(IMAGE_SIZE),
                T.ToTensor(),
                normalize
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # A imagem no MedMNIST é um array NumPy HxWxC
        img_array = self.images[idx]
        label = int(self.labels[idx])
        
        # Converter para PIL Image para aplicar as transformações do Torchvision
        img = Image.fromarray(img_array).convert('RGB')
        
        if self.transforms:
            img = self.transforms(img)
            
        return img, label
