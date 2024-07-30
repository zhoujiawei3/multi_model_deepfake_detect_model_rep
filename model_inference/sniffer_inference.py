import torch
import numpy as np
import os
import pandas as pd
from torch.utils.data import  DataLoader
from transformers import AutoProcessor, InstructBlipForConditionalGeneration,InstructBlipProcessor,InstructBlipConfig
import torchvision.transforms as transforms
from peft import LoraConfig, get_peft_model
import argparse
import glob
from tqdm import tqdm
from peft import PeftModel, PeftConfig

from transformers import ViTImageProcessor, ViTForImageClassification,AutoModel,AutoConfig,PretrainedConfig
from PIL import Image
from torch import nn
import random
from torchvision import datasets
from transformers import AutoTokenizer
from nltk.translate.meteor_score import meteor_score

from lavis.models import load_model_and_preprocess
parser = argparse.ArgumentParser(description="Fine-Tune BLIP-2 for Diffusion-based Generated Images Detection.")
parser.add_argument('--device', default='1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
opt = parser.parse_args()
device = torch.device(f"cuda:"+opt.device if torch.cuda.is_available() else "cpu")

model_blip2, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna13b", is_eval=True, device=device)
model_blip2=model_blip2.to(device)

image=Image.open('/data1/zhoujiawei/DiFF_mix_new/train/fake/FE_CoDiff_1.png')
instruction_txt = "Which type of deepfake the image is? Which deepfake method is used to generate it?"
image = vis_processors["eval"](image).unsqueeze(0).to(device)
generated_ids = model_blip2.generate({"image": image, "prompt":instruction_txt})
out_text=generated_ids
print("result_text:",out_text)