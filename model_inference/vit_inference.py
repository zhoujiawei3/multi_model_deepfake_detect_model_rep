import torch
import numpy as np
import os
import pandas as pd
from torch.utils.data import  DataLoader
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torchvision.transforms as transforms
from peft import LoraConfig, get_peft_model
import argparse
import glob
from tqdm import tqdm
from peft import PeftModel, PeftConfig
from datasets import load_dataset 
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
from torch import nn
import random
from torchvision import datasets
from transformers import AutoTokenizer
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
parser = argparse.ArgumentParser(description="Fine-Tune BLIP-2 for Diffusion-based Generated Images Detection.")
parser.add_argument('--device', default='1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
opt = parser.parse_args()
device = torch.device(f"cuda:"+opt.device if torch.cuda.is_available() else "cpu")




processor_vit = ViTImageProcessor.from_pretrained('/data1/zhoujiawei/hg_hub/models--google--vit-huge-patch14-224-in21k')
model_vit = torch.load('../部分参数微调_huge_vit_DiFF/20/pytorch_model.bin').to(device) 
    # model_vit.classifier=nn.Linear(in_features =1280, out_features = 2)
    # model.config.problem_type == "single_label_classification"
model_vit.eval()


image=Image.open('/data1/zhoujiawei/DiFF_mix_new/train/fake/FE_CoDiff_1.png')
inputs_vit = processor_vit(images=image, return_tensors="pt").to(device)
outputs_vit = model_vit(**inputs_vit)
logits = outputs_vit.logits
predicted_class_idxs= logits.argmax(-1)
#0是fake,1是real
print(predicted_class_idxs)