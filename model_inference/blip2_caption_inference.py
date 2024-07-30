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
parser.add_argument('--continue_path', type=str, default='/home/zhoujiawei/blip2_finetune/oldcaption_epoch_24',
                        help='Path to save trained model.')

opt = parser.parse_args()
device = torch.device(f"cuda:"+opt.device if torch.cuda.is_available() else "cpu")

config = PeftConfig.from_pretrained(opt.continue_path)
model_blip2 = Blip2ForConditionalGeneration.from_pretrained(config.base_model_name_or_path, load_in_8bit=True, device_map=device)
model_blip2 = PeftModel.from_pretrained(model_blip2, opt.continue_path)
processor_blip2 = AutoProcessor.from_pretrained("/data/MLLM_models/blip2-opt-2.7b/")
# for name, param in model_blip2.named_parameters():
#     if "lora" in name:
#         param.requires_grad = True
#     else:
#         param.requires_grad = False
# model_blip2.print_trainable_parameters()
model_blip2 = model_blip2.eval()

image=Image.open('/data1/zhoujiawei/DiFF_mix_new/train/fake/FE_CoDiff_1.png')
inputs_blip2 = processor_blip2(images=image, return_tensors="pt").to(device)
outputs_blip2 = model_blip2.generate(pixel_values=inputs_blip2['pixel_values'], max_length=20)
out_text=processor_blip2.batch_decode(outputs_blip2, skip_special_tokens=True)[0]
#形如: Fake FE CoDiff
print("result_text:",out_text)
