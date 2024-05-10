import multiprocessing
import torch
import torchvision.transforms as transforms
from torchvision.datasets.vision import VisionDataset 
from typing import List, Tuple,Any
from typing import Any, Callable, Optional, Tuple
import  pickle
import cv2
import numpy as np

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
class MyDataset(VisionDataset):
    def __init__(
        self,
        data_path: str,        
        transform: Optional[Callable]=None,
        normalize: Optional[Callable]=None,
    ) -> None:
        # super().__init__(root, transform=transform, target_transform=target_transform)
        self.transform = transform
        self.normalize = normalize
        with open(data_path, 'rb') as fo:
            dict = pickle.load(fo, encoding='latin-1')
        imgs1 = dict['imgs1'.encode('utf-8')]
        imgs2 = dict['imgs2'.encode('utf-8')]

        self.imgs1 = []
        self.imgs1_tensor = []
        self.imgs1_tensor_nor = []   
        self.imgs2 = []  
        self.imgs2_tensor = []
        self.imgs2_tensor_nor = []
        for j in range(len(imgs1)):
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img1 =  imgs1[j]
            img2 =  imgs2[j]

            if self.transform is not None :
                # plt.imshow(transforms.ToTensor()(img.astype(np.uint8)).permute(1,2,0))
                img1 = self.transform(img1) 
                img2 = self.transform(img2) 
            else:
                img1 = torch.tensor(img1.astype(np.float32))
                img2 = torch.tensor(img1.astype(np.float32))

            self.imgs1_tensor.append(img1)
            self.imgs2_tensor.append(img2)

            if self.normalize is not None :
                img1 = self.normalize(img1) 
                img2 = self.normalize(img2) 

            self.imgs1_tensor_nor.append(img1)
            self.imgs2_tensor_nor.append(img2)

            self.imgs1.append(np.array(imgs1[j]))
            self.imgs2.append(np.array(imgs2[j]))
       
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        index1 = index*np.random.randint(0,100)%len(self.imgs1)        
        index2 = index*np.random.randint(0,100)%len(self.imgs2)        
        return self.imgs1_tensor_nor[index1],self.imgs1_tensor[index1],self.imgs1[index1], \
               self.imgs2_tensor_nor[index2],self.imgs2_tensor[index2],self.imgs2[index2]
    def __len__(self) -> int:
        return 16*200
        # return len(imgs1)

def get_data_loader(binTrain,binEval,batch_size):
    # 数据增强
    original_transforms_train = transforms.Compose([
        transforms.Resize((128,128)),
        # transforms.RandomCrop(size=64, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        # transforms.Normalize([0.4479157, 0.3059063, 0.24577492], [0.35524514, 0.2552989, 0.21375151])
    ])
    original_transforms_eval = transforms.Compose([
        transforms.Resize((128,128)),
        # transforms.RandomCrop(size=64, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        # transforms.Normalize([0.44207954, 0.33953816, 0.29850873], [0.3788479, 0.30274758, 0.26969728])
    ])
    # 数据集
    train_normalize = transforms.Normalize([0.604163, 0.5305837, 0.5129956], [0.27273604, 0.27319935, 0.27248213])
    eval_normalize = transforms.Normalize([0.604163, 0.5305837, 0.5129956], [0.27273604, 0.27319935, 0.27248213])

    dataset_train = MyDataset(data_path=binTrain, transform=original_transforms_train, normalize=train_normalize)
    dataset_eval = MyDataset(data_path=binEval, transform=original_transforms_eval, normalize=eval_normalize)
    # loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2,pin_memory=True,prefetch_factor = batch_size)
    loader_eval = torch.utils.data.DataLoader(dataset_eval, batch_size=batch_size, shuffle=True, num_workers=2,pin_memory=True,prefetch_factor = batch_size)

    return loader_train,dataset_train,loader_eval,dataset_eval
