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
        #使用load_dataset的时候
        # item = self.dataset[idx] #image:路径 text：Real or Fake
        #使用parser_data的时候
        item = self.dataset.loc[idx]
        # 0print(item.keys())
        image=item["image_bytes"]

        label=item["label"]
        image=Image.open(io.BytesIO(image))
        image=np.array(image,dtype=int)
        label=np.array(label,dtype=int)
        image_path=item["image_path"]
        if label == 1:
            deepfake_way = 'real'
            deepfake_method = 'real'
            description= f'Real {deepfake_way} {deepfake_method}'
        else:
            deepfake_way = image_path.split('/')[-1].split('_')[0]
            deepfake_method = image_path.split('/')[-1].split('_')[1]
            captionmap={'T2I':'Text-to-Image','FS':'Face Swap','I2I':'Image-to-Image','FE':'Face Editing'}

            description= f"The image is fake. It is a {captionmap[deepfake_way]} deepfake. It is generated by {deepfake_method}."

        # deepfake_way=np.array(deepfake_way)
        # deepfake_method=np.array(deepfake_method)
        description=np.array(description)
        return image,label,description
       