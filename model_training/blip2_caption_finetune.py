import torch
import numpy as np
import os
import pandas as pd
from dataset.dataset_caption_finetune import ImageCaptioningDataset
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


# Set random seed for PyTorch

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True  #这将确保每次运行程序时，对于相同的输入，卷积等操作的输出始终是一样的。这对于调试和复现结果非常有用。
    torch.backends.cudnn.benchmark = False #禁用cudnn的自动优化。当输入数据的大小变化不大时，启用此选项可以加速程序。但是，如果输入数据的大小经常变化，那么启用此选项可能会导致程序变慢。因此，这里选择禁用它。

# Set random seed for NumPy

np.random.seed(RANDOM_SEED)

def collate_fn(batch):
    # pad the input_ids and attention_mask
    image,label,description=zip(*batch)
    images = np.stack(image, axis=0)
    labels = np.array(label)
    descriptions = np.array(description)

    return images,labels,descriptions

def print_qformer_trainable_parameters(model):
    qformer_params = [name for name, param in model.named_parameters() if 'qformer' in name and param.requires_grad]
    print("trainable parameter in Q-Former:")
    for param in qformer_params:
        print(param)

# Main Body
if __name__ == "__main__":

    #device 变成cuda 0
    parser = argparse.ArgumentParser(description="Fine-Tune BLIP-2 for Diffusion-based Generated Images Detection.")

    parser.add_argument('--dataset', default='/data1/zhoujiawei/deepfake-and-real-images/datasets--Hemg--deepfake-and-real-images/data', type=str,
                        help='Path to the training CSV file')
    parser.add_argument('--epochs', default=80, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--lr', default=5e-5, type=float,
                        help='The learning rate for training (default: 5e-5).')
    parser.add_argument('--continue_path', type=str, default='',
                        help='Path to save trained model.')
    parser.add_argument('--save_path', type=str, default='./DIFF_blip2_caption_last',
                        help='Path to save trained model.')
    parser.add_argument('--device', default='1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    
    opt = parser.parse_args()

    device = torch.device(f"cuda:"+opt.device if torch.cuda.is_available() else "cpu")
    #模型载入
        #blip2

    
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)

    if opt.continue_path=='':
        #load model
        model_blip2 = Blip2ForConditionalGeneration.from_pretrained("/data/MLLM_models/blip2-opt-2.7b/" ,load_in_8bit=True, device_map=device)
        processor_blip2 = AutoProcessor.from_pretrained("/data/MLLM_models/blip2-opt-2.7b/")

        #train LLM with lora
        config_lora = LoraConfig(
            r=64,
            lora_alpha=16,
            lora_dropout=0.05,
            bias='none',
            target_modules=['q_proj', 'k_proj'] # 这里因为只有LLM中有'q_proj', 'k_proj'，因此Qformer是没有被优化的
        )
        model_blip2 = get_peft_model(model_blip2, config_lora)
        model_blip2.print_trainable_parameters()


        # train qformer

        # for name, param in model_blip2.named_parameters():
        #     if "qformer" in name:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False
        # total_params = sum(param.numel() for param in model_blip2.parameters())
        # trainable_params = sum(param.numel() for param in model_blip2.parameters() if param.requires_grad)

        # print(f"Total Parameters: {total_params}")
        # print(f"Trainable Parameters: {trainable_params}")


    else:
        #only lora continue training finished
        config = PeftConfig.from_pretrained(opt.continue_path)
        model_blip2 = Blip2ForConditionalGeneration.from_pretrained(config.base_model_name_or_path, load_in_8bit=True, device_map=device)
        model_blip2 = PeftModel.from_pretrained(model_blip2, opt.continue_path)
        processor_blip2 = AutoProcessor.from_pretrained("/data/MLLM_models/blip2-opt-2.7b/")
        for name, param in model_blip2.named_parameters():
            if "lora" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        model_blip2.print_trainable_parameters()
    optimizer = torch.optim.Adam(model_blip2.parameters(), lr=opt.lr)

    #use vit to only caption fake image, optional
    processor_vit = ViTImageProcessor.from_pretrained('/data1/zhoujiawei/hg_hub/models--google--vit-huge-patch14-224-in21k')
    model_vit = torch.load('../部分参数微调_huge_vit_DiFF/20/pytorch_model.bin').to(device) 
    model_vit.eval()


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

    #train
    continueFrom_epoch=0
    
    for epoch in range(continueFrom_epoch,opt.epochs+continueFrom_epoch):
        if epoch!=0:
            #test
            nb_test=len(test_dataloader)
            pbar_test = enumerate(test_dataloader)
            pbar_test = tqdm(pbar_test, total=nb_test)
            print(("\n"+ '%15s' * 5) %('Epoch','METEOR','mean_METEOR','ROUGE','mean_ROUGE'))

            result_texts=[]
            label_texts=[]
            # mean_meteor_score=0
            mean_rouge_score=0
            result = []
            right=0
            wrong_number=0
            
            for idx_test, (images,labels,descriptions) in pbar_test:
                inputs_vit = processor_vit(images=images, return_tensors="pt").to(device)
                outputs_vit = model_vit(**inputs_vit)
                logits = outputs_vit.logits
                predicted_class_idxs= logits.argmax(-1)
                for i,predicted_class_idx in enumerate(predicted_class_idxs.cpu()):
                    if predicted_class_idx==labels[i]:
                        right+=1
                    result.append(predicted_class_idx)
                    if predicted_class_idx==0:
                        wrong_number=wrong_number+1
                        images_fake=images[i]
                        descriptions_fake=descriptions[i]

                        #预处理

                        #给descriptions添加一个batch维度
                        encoding = processor_blip2(images=images_fake, padding='max_length', return_tensors='pt')
                        encoding = {k: v.squeeze() for k, v in encoding.items()}
                        pixel_values = encoding["pixel_values"].to(device)
                        pixel_values = pixel_values.unsqueeze(0)
                        # print(pixel_values.shape)
                        
                        generated_ids = model_blip2.generate(pixel_values=pixel_values, max_length=30)
                        out_text=processor_blip2.batch_decode(generated_ids, skip_special_tokens=True)[0]
                        # print("out_text:",out_text)
                        result_texts.append(out_text)
                        label_texts.append(descriptions_fake)
                        if idx_test%10==0:
                            print("result_text:",out_text)
                            print("label_text:",descriptions_fake)

                        # calculate BLEU
                        # bleu_score = corpus_bleu([[descriptions_fake.split()]], [out_text.split()] )
                        # mean_bleu_score=(wrong_number*mean_bleu_score+bleu_score)/(wrong_number+1)

                        # calculate METEOR
                        # meteor = meteor_score([descriptions_fake.split()], out_text.split())
                        # mean_meteor_score=(wrong_number*mean_meteor_score+meteor)/(wrong_number+1)

                        #calculate ROUGE
                        rouge = Rouge()
                        rouge_scores = rouge.get_scores(out_text, descriptions_fake)[0]['rouge-l']['f']
                        mean_rouge_score=(wrong_number*mean_rouge_score+rouge_scores)/(wrong_number+1)


                        # s=('%10s' * 1 + '%10.4g' * 4) % (epoch, meteor, meteor_scoremean_,rouge_scores,mean_rouge_score)
                        s=('%10s' * 1 + '%10.4g' * 2) % (epoch,rouge_scores,mean_rouge_score)
                        pbar_test.set_description(s)
            print(len([x for x in result[:5784] if x==0]))
            print(len([x for x in result[:5784]if x==1]))
            print(len([x for x in result[5784:]if x==0]))
            print(len([x for x in result[5784:] if x==1]))
            print(f"TNR = {len([x for x in result[:5784] if x==0])/ 5784}")
            print(f"FPR = {len([x for x in result[:5784] if x==1])/ 5784}")
            print(f"FNR = {len([x for x in result[5784:] if x==0])/ 5784}")
            print(f"TPR = {len([x for x in result[5784:] if x==1])/ 5784}")
            right_rate=right/nb_test
            print (epoch,"right rate:",right_rate)
            # print("mean_meteor_score:",mean_meteor_score)
            print("mean_rouge_score:",mean_rouge_score)

            with open(f"{opt.save_path}/test.txt","a") as f:
                f.write(f"epoch:{epoch} mean_rouge:{mean_rouge_score}")
                f.write(f"right rate:{right_rate} ")
                f.write( f" TNR:{len([x for x in result[:5784] if x==0])/ 5784} FPR: {len([x for x in result[:5784] if x==1])/ 5784} FNR:{len([x for x in result[5784:] if x==0])/ 5784} TPR:{len([x for x in result[5784:] if x==1])/ 5784}\n")
            
        #real_train
        nb=len(train_dataloader)
        pbar = enumerate(train_dataloader)
        pbar = tqdm(pbar, total=nb)
        print("start to train\n")
        print(('\n' + '%10s' * 2) % ('Epoch','loss'))
        for idx, (images,labels,descriptions) in pbar:
            #use vit to remove real image，only caption fake image, could remove
            inputs_vit = processor_vit(images=images, return_tensors="pt").to(device)
            outputs_vit = model_vit(**inputs_vit)
            logits = outputs_vit.logits
            predicted_class_idxs= logits.argmax(-1)
            bool_idxs = (predicted_class_idxs == 0).flatten().cpu()


            images_fake=images[bool_idxs]
            descriptions_fake=descriptions[bool_idxs]
            
            descriptions_fake=descriptions_fake.tolist()
            fake_number = images_fake.shape[0]
            inputs = processor_blip2(images=images_fake, text=descriptions_fake,padding=True, return_tensors='pt').to(device)

            outputs = model_blip2(input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        labels=inputs["input_ids"])
            
            if idx%150==0:
                generated_output = model_blip2.generate(
                                pixel_values=inputs["pixel_values"], max_length=20
                            )
                print(f"caption: {processor_blip2.batch_decode(generated_output, skip_special_tokens=True)[0]}")

            loss = outputs.loss #通常来说是Cross-Entropy Loss

            s=('%10s' * 1 + '%10.4g' * 1) % (epoch, loss)
            pbar.set_description(s)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        checkpoint_dir=os.path.join(opt.save_path, f"epoch_{epoch}/")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        model_blip2.save_pretrained(checkpoint_dir)
   
   
