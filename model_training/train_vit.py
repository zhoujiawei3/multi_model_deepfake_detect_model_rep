from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests
from torch import nn
import numpy as np
import random
import torch
from dataset.dataset_vit import ImageCaptioningDataset
from torch.utils.data import  DataLoader
from torchvision import datasets
import glob
import pandas as pd
from tqdm import tqdm
import os
from datasets import load_dataset 
def get_data(batch_size,label_num,path):
    num=int(batch_size/len(label_num))
    data=[]
    for sub_data in label_num.values():
        if num<=len(sub_data):
            data.extend(random.sample(sub_data,num))
        else:
            data.extend(sub_data)

    images=[]
    labels=[]
    for p,label in data:
        try:  
            img_path="{}{}".format(path,p)
            img = Image.open(img_path)
            img2=np.array(img)
            img.close()
            if len(img2.shape)!=3:
                continue     
            images.append(img2)
            labels.append(label)
        except:
            continue
    return images,np.array(labels)
def no_contain(name,freeze_layer):
    for s in freeze_layer:
        if s in name:
            return False
    return True
def parser_data(path):
    with open(path,encoding="utf-8") as f:
        lines=[ eval(s.strip()) for s in f.readlines()]
    label_num={}
    for p,label in lines:
        if label not in label_num:
            label_num[label]=[]
        label_num[label].append([p,label])
    return label_num
def cal_right_rate(val_data):
    right=0
    images=val_data.data
    labels=val_data.targets
    data=random.sample([s for s in zip(images,labels)],500)
    images,labels=zip(*data)
    count=len(labels)
    for i,val_image in  enumerate(images):
        inputs = processor(images=val_image, return_tensors="pt").to("cuda:5") 
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        if predicted_class_idx==labels[i]:
            right+=1
    return right/count


def cal_paras(model):
    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0
    for param in model.parameters():
        mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
        Total_params += mulValue  # 总参数量
        if param.requires_grad:
            Trainable_params += mulValue  # 可训练参数量
        else:
            NonTrainable_params += mulValue  # 非可训练参数量
    print(f'Total params: {Total_params}')
    print(f'Trainable params: {Trainable_params}')
    print(f'Non-trainable params: {NonTrainable_params}')
def collate_fn(batch):
    image,label=zip(*batch)
    images = np.stack(image, axis=0)
    labels = np.array(label)

    return images, labels
    
import torchvision.transforms as transforms

transform = transforms.Compose( [transforms.ToTensor()])
#只有100类
# parquet_files = glob.glob("/data1/zhoujiawei/DiFF_mix/train/parquet")
# dataframes = [pd.read_parquet(file) for file in parquet_files]
#load dataset 
        #train
parquet_files = glob.glob("/data1/zhoujiawei/DiFF_mix_new/train/parquet")
dataframes = [pd.read_parquet(file) for file in parquet_files]
tr_dataset= pd.concat(dataframes, ignore_index=True)
train_dataset = ImageCaptioningDataset(tr_dataset)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32, collate_fn=collate_fn)

    #test
parquet_files_test = glob.glob("/data1/zhoujiawei/DiFF_mix_new/test/parquet")
dataframes_test = [pd.read_parquet(file) for file in parquet_files_test]
te_dataset= pd.concat(dataframes_test , ignore_index=True)
test_dataset = ImageCaptioningDataset(te_dataset)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1, collate_fn=collate_fn)



# train_dataset = datasets.cifar.CIFAR100(root='cifar100', train=True,transform=transform, download=True)
# train_dataset.data=np.array(train_dataset.data)
# trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
# test_dataset = datasets.cifar.CIFAR100(root='cifar100', train=False,transform=transform, download=True)

processor = ViTImageProcessor.from_pretrained('/data1/zhoujiawei/hg_hub/models--google--vit-huge-patch14-224-in21k')
model = ViTForImageClassification.from_pretrained('/data1/zhoujiawei/hg_hub/models--google--vit-huge-patch14-224-in21k')
# processor = ViTImageProcessor.from_pretrained('/home/zhoujiawei/vit-simple/vitbase')
# model = ViTForImageClassification.from_pretrained('/home/zhoujiawei/vit-simple/vitbase')

train_layer=["9","10","11"]
print("只调MLP")
if not os.path.exists("部分参数微调_huge_vit_DiFF/"):
    os.makedirs("部分参数微调_huge_vit_DiFF/")
for  name,p in model.named_parameters():
    if no_contain(name,train_layer):
        #不在train_layer里面的 我都可以冻住
        p.requires_grad=False
    else:
        p.requires_grad=True
#分类层：改变模型结构
model.classifier=nn.Linear(in_features =1280, out_features = 2)
model.to("cuda:5") 
cal_paras(model)
model.config.problem_type == "single_label_classification"
#改对应的配置
model.num_labels=2
batch_size=256
learning_rate=1e-3
epochs=100
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
step=0
print("ready-to_train")
for epoch in range(epochs):
    
    nb=len(train_dataloader)
    pbar = enumerate(train_dataloader)
    pbar = tqdm(pbar, total=nb)
    print(('\n' + '%10s' * 3) % ('Epoch', 'gpu_mem','loss'))
    for batch_idx, (images, labels) in pbar:
        if step%700==0 or step==0:
            model.eval()
            nb_test=len(test_dataloader)
            pbar_test = enumerate(test_dataloader)
            pbar_test = tqdm(pbar_test, total=nb_test)
            right=0
            result = []
            for batch_idx_test, (images_test, labels_test) in pbar_test:
                inputs_test = processor(images=images_test, return_tensors="pt").to("cuda:5") 
                outputs_test = model(**inputs_test)
                logits = outputs_test.logits
                predicted_class_idx= logits.argmax(-1).item()
                if predicted_class_idx==labels_test:
                    right+=1
                result.append(predicted_class_idx)
            print(len([x for x in result[:5784] if x==0]))
            print(len([x for x in result[:5784]if x==1]))
            print(len([x for x in result[5784:]if x==0]))
            print(len([x for x in result[5784:] if x==1]))
            print(f"TNR = {len([x for x in result[:5784] if x==0])/ 5784}")
            print(f"FPR = {len([x for x in result[:5784] if x==1])/ 5784}")
            print(f"FNR = {len([x for x in result[5784:] if x==0])/ 5784}")
            print(f"TPR = {len([x for x in result[5784:] if x==1])/ 5784}")
            
            right_rate=right/nb_test
            print (step,"准确率",right_rate)
            #输出到一个text文件
            
            with open("部分参数微调_huge_vit_DiFF/部分参数微调_huge_vit_DiFF.txt","a") as f:
                f.write(f"epoch{epoch} step:{step} 准确率:{right_rate}")
                f.write( f" TNR:{len([x for x in result[:5784] if x==0])/ 5784} FPR: {len([x for x in result[:5784] if x==1])/ 5784} FNR:{len([x for x in result[5784:] if x==0])/ 5784} TPR:{len([x for x in result[5784:] if x==1])/ 5784}\n")
        step+=1
        model.train()
        # images=np.array(255*images,dtype=int)
        # print(images.shape)
        inputs = processor(images=images, return_tensors="pt").to("cuda:5")
        # labels=np.array(labels,dtype=int) 
        
        # print(labels.shape)
        labels = torch.from_numpy(labels).to("cuda:5").long()
        #print(inputs.shape)
        #print(labels.shape)
        loss= model(**inputs,return_dict=False,labels=labels)[0]
        optimizer.zero_grad() # 梯度置零，因为反向传播过程中梯度会累加上一次循环的梯度
        # print (loss)
        loss.backward() # loss反向传播
        optimizer.step() # 反向传播后参数更新 
        # print(f"torch.cuda.memory_reserved()={torch.cuda.memory_reserved()}")
        mem='%.3gM' % (torch.cuda.memory_reserved() / 1E6 if torch.cuda.is_available() else 0)
        s=('%10s' * 2 + '%10.4g' * 1) % (epoch, mem,loss.detach().data.cpu().numpy())
        pbar.set_description(s)
        
    torch.save(model, f"部分参数微调_huge_vit_DiFF/pytorch_model_epoch{epoch}.bin")
#torch.save(model, "my_model\\pytorch_model.bin") 
