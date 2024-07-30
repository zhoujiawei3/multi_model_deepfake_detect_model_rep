from torch.utils.data import Dataset
import PIL.Image as Image
import io
import numpy as np
# Image Captioning Dataset Class

class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # item = self.dataset.loc[idx] #image:路径 text：Real or Fake
        item = self.dataset.loc[idx]
        # encoding = self.processor(images=Image.open(item["image"]), padding="max_length", return_tensors="pt")#max_length表示如果图像的尺寸小于预处理器设定的最大长度，那么会在图像周围填充0（或其他值）以达到最大长度。这是为了确保所有处理后的图像都有相同的尺寸 pt表示pytorch张量
        image=item["image_bytes"]
        image=Image.open(io.BytesIO(image))
        image=np.array(image,dtype=int)
        # print(image.max())    
        label=item["label"]
        label=np.array(label,dtype=int)
        # remove batch dimension
        
        return image,label