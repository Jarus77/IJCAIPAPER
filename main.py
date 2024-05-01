import json
import csv
import torch
import torchvision
from torch.nn import CrossEntropyLoss

from transformers import LayoutLMForTokenClassification
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import scipy
from scipy.interpolate import interp1d
import numpy as np
from pypdf import PdfReader, PdfWriter
from pypdf.annotations import Rectangle, FreeText
from pdf_annotate import PdfAnnotator, Appearance, Location
import PyPDF2
import fpdf
from fpdf import FPDF
from pathlib import Path
from uuid import uuid4
# from label_studio_sdk import Client
# from label_studio_sdk import Project
# from label_studio_sdk.data_manager import Column, Filters, Operator, Type
from datetime import datetime
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import re
import sys
import time
import os 
from os.path import join
from xml.dom import minidom
import json


from PIL import Image
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertConfig, BertModel, BertPreTrainedModel, get_linear_schedule_with_warmup, AdamW, BertTokenizerFast
from torch.nn import LayerNorm as BertLayerNorm
import torch
import torchvision
import json
import os
from PIL import Image, ImageDraw, ImageFont
import cv2
import matplotlib.pyplot as plt
import os
import json
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import numpy as np

class CordDataset(Dataset):
    def __init__(self, examples, tokenizer, labels, pad_token_label_id):
        max_seq_length = 512
        features = convert_examples_to_featuresz(
            examples,
            labels,
            max_seq_length,
            tokenizer,
            cls_token_at_end=False,
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=False,
            pad_on_left=False,
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=0,
            pad_token_label_id=pad_token_label_id,
        )

        self.features = features
        self.all_input_ids = torch.tensor(
            [f.input_ids for f in features], dtype=torch.long
        )
        self.all_input_mask = torch.tensor(
            [f.input_mask for f in features], dtype=torch.long
        )
        self.all_segment_ids = torch.tensor(
            [f.segment_ids for f in features], dtype=torch.long
        )
        self.all_label_ids = torch.tensor(
            [f.label_ids for f in features], dtype=torch.long
        )
        self.all_bboxes = torch.tensor([f.boxes for f in features], dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return (
            self.all_input_ids[index],
            self.all_input_mask[index],
            self.all_segment_ids[index],
            self.all_label_ids[index],
            self.all_bboxes[index],
        )

class InputFeatures(object):
    def __init__(
        self,
        input_ids,
        input_mask,
        segment_ids,
        label_ids,
        boxes
    ):
        assert (
            0 <= all(boxes) <= 1000
        ), "Error with input bbox ({}): the coordinate value is not between 0 and 1000".format(
            boxes
        )
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.boxes = boxes

def convert_examples_to_featuresz(
    examples,
    label_list,
    max_seq_length,
    tokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    cls_token_box=[0, 0, 0, 0],
    sep_token_box=[1000, 1000, 1000, 1000],
    pad_token_box=[0, 0, 0, 0],
    pad_token_segment_id=0,
    pad_token_label_id=-1,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
):
    label_map = {i: label for i, label in enumerate(label_list)}
    # print(label_map)
    features = []
    for i in range(len(examples[0])):
        width, height = 1000, 1000
        words = examples[0]
        labels = examples[1]
        boxes = examples[2]

        tokens = []
        token_boxes = []
        label_ids = []
        for word, label, box in zip(
            words[i], labels[i], boxes[i]
        ):
            # print(word, label)
            if len(word) < 1:
              continue
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            token_boxes.extend([box] * len(word_tokens))
            label_ids.extend(
                [label] + [pad_token_label_id] * (len(word_tokens) - 1))
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            token_boxes = token_boxes[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

        tokens += [sep_token]
        token_boxes += [sep_token_box]
        label_ids += [pad_token_label_id]
        if sep_token_extra:
            tokens += [sep_token]
            token_boxes += [sep_token_box]
            label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            token_boxes += [cls_token_box]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            token_boxes = [cls_token_box] + token_boxes
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = (
                [0 if mask_padding_with_zero else 1] * padding_length
            ) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
            token_boxes = ([pad_token_box] * padding_length) + token_boxes
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length
            token_boxes += [pad_token_box] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(token_boxes) == max_seq_length
        print(label_ids)
        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_ids=label_ids,
                boxes=token_boxes,
            )
        )
    return features


def normalize_bbox(bbox, width, height):
     return [
         int(1000 * (bbox[0] / width)),
         int(1000 * (bbox[1] / height)),
         int(1000 * (bbox[2] / width)),
         int(1000 * (bbox[3] / height)),
     ]

def CORD_test(f):
  labeled=0
  ####################
  # cord Dataset
  ####################

  # Process the Raw Labelled cord Data
  # Data consisting of 112 annotated medical documents
#   directory = 'CORD/train/json'
  
  # iterate over files
  words = []
  bbox = []
  labels = []
  # for filename in sorted(os.listdir(directory)):
  #     f = os.path.join(directory, filename)
  #     # checking if it is a file
  if os.path.isfile(f):
    dataraw = open(f)
    jdata = json.load(dataraw)
    height = 1 # jdata['meta']['image_size']['height']
    width = 1 # jdata['meta']['image_size']['width']
    wordL = []
    boxes = []
    label = []
    for page in jdata['pages']:
        for block in page['blocks']:
            for line in block['lines']:
                for word in line['words']:
                    geometry = word['geometry']
                    xmin, ymin = geometry[0]
                    xmax, ymax = geometry[1]
                    txt = word['value']
                    box = [xmin, ymin, xmax, ymax]
                    box = normalize_bbox(box, width=width, height=height) 
                    wordL.append(txt)
                    boxes.append(box)  
                    if("category" in word.keys()):
                        label.append(word["category"])
                        labeled = 1
                    else:
                        label.append("OTHERS")
    if labeled==1:
        words.insert(0,wordL) 
        bbox.insert(0,boxes) 
        labels.insert(0,label)
        labeled = 0
    else:
        words.append(wordL) 
        bbox.append(boxes) 
        labels.append(label)
  return words,labels,bbox 

def preprossing(label):
    for i in range(len(label)):
        for j in range(len(label[i])):
            if (label[i][j] == "PATIENT_NAME"):
                label[i][j] = 1
            elif (label[i][j] == 'PATIENT_ID'):
                label[i][j] = 0   
            elif (label[i][j]=='LOCATION'):
                label[i][j] = 2
            else:
                label[i][j] = 3
    return label    

def project(a, A,B,C,D):
    scale = (D-C)/(B-A)
    offset = -A*(D-C)/(B-A) + C
    return a*scale + offset

def translate(bbox, A,B,C,D):
    interpX = interp1d([0, A], [0, C])
    interpY = interp1d([0, B], [0, D])
    tempbbox = []
    for box in bbox:
        newbox =[]
        newbox.append(float(interpX(box[0])))
        newbox.append(float(interpY(box[1]))-2)
        newbox.append(float(interpX(box[2])))
        newbox.append(float(interpY(box[3])))
        tempbbox.append(newbox) 
    return tempbbox

def denormalize(bbox, W, H):
    tempbbox = []
    for box in bbox:
        newbox =[]
        newbox.append(W*box[0]/1000)
        newbox.append(H*box[1]/1000)
        newbox.append(W*box[2]/1000)
        newbox.append(H*box[3]/1000)
        tempbbox.append(newbox)
    return tempbbox

def get_model_preds(pdf_path, json_path):
    token_test, label_test, boxe_test = CORD_test(json_path)
    label_test=preprossing(label_test)

    demo = np.array([token_test, label_test, boxe_test])
    print(len(demo))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from collections import Counter
    # all_labels = [item for sublist in demo[1] for item in sublist]
    # Counter(all_labels)
    # labels = list(set(all_labels))
    labels=['PATIENT_ID','PATIENT_NAME','LOCATION', 'OTHERS']
    num_labels = len(labels)
    label_map = {i: label for i, label in enumerate(labels)}
    # print(label_map)
    pad_token_label_id = CrossEntropyLoss().ignore_index

    from transformers import LayoutLMTokenizer
    from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
    max_seq_length = 512
    tokenizer = tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")

    demo_ = CordDataset(demo, tokenizer, labels, pad_token_label_id)
    demo_sampler = SequentialSampler(demo_)
    demo_dataloader = DataLoader(demo_, 
                                sampler=demo_sampler,
                                batch_size=1)   
    # for batch in demo_dataloader:
    #     print(0)
    #     print(batch)
    #     print("OVER")
    #     break

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LayoutLMForTokenClassification.from_pretrained("./Paths/model_weights/pytorch_model.bin", config='./Paths/model_weights/config.json', local_files_only=True)
    # model.load_state_dict(torch.load("./model_weights.pth"))
    model.to(device)
    # print(demo_dataloader.sampler)
    
    preds = None
    out_label_ids = None
    # print("Here1")
    model.eval()
    # print("Here3")
    for batch in tqdm(demo_dataloader, desc="Evaluating"):
        # print(batch)
        with torch.no_grad():
            # print("Here0")
            input_ids = batch[0].to(device)
            bbox = batch[4].to(device)
            attention_mask = batch[1].to(device)
            token_type_ids = batch[2].to(device)
            labels = batch[3].to(device)
            # labels = batch[3].to(device)
            # forward pass
            outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids)
            # get the loss and logits
            # tmp_eval_loss = outputs.loss
            logits = outputs.logits
            if preds is None:
                # print("Here")
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()

            else:
                # print("Here2")
                # print("PREDS")
                # print(preds)
                out_label_ids = np.append(
                    out_label_ids, labels.detach().cpu().numpy(), axis=0
                )

    # compute average evaluation loss
    preds = np.argmax(preds, axis=2)
    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]
    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])
        # print(len(preds_list[i]))
    # # from seqeval.metrics import accuracy_score as acc
    # # fm_test_acc = acc(out_label_list, preds_list)
    # # # print()
    # # print(fm_test_acc)
    print(preds_list)
    return demo, preds_list


def anonymize(pdf_path, new_pdf_path, bboxes, labels, words):
    W, H = 1190, 1683
    w,h = 600,840
    new_text = "xxxxxxxxxxxxxxx"
    reader = PdfReader(pdf_path)
    page = reader.pages[0]
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    pdf_file = PyPDF2.PdfReader(pdf_path)
    page = pdf_file.pages[0]
    print(page)
    text = page.extract_text()
    new_text = "xxxxxxxxxxxxxxx"
    for bbox, label, word in zip(bboxes, labels, words):
        if label=='PATIENT_ID' or label=='PATIENT_NAME':
            print(word)
            text = text.replace(word, new_text)
        else:
            text = text.replace(word, new_text, 1)
    pdf = FPDF()
    text2 = text.encode('utf-8').decode('utf-8')
    lines_per_page = 4000
    text_pages = [text2[i:i+lines_per_page] for i in range(0, len(text2), lines_per_page)]
    pdf.set_font("Arial", size=12)
    for text_page in text_pages:
        pdf.add_page()
        pdf.multi_cell(200, 10, txt=text_page, align="L")
    for page in pdf_file.pages[1:]:
        pdf.add_page()
        pdf.multi_cell(200, 10, txt=page.extract_text(), align="L")
    # output_path = os.path.join(new_pdf_path, f'MaskedPDF{idx}')
    pdf.output(new_pdf_path, 'F')

    
        
def img_to_pdf(img_dir, pdf_path, idx):
    images = []
    slash = '/'
    for fname in os.listdir(img_dir):
        # fname = slash + fname
        img_path = os.path.join(img_dir, fname)

        img = Image.open(img_path)
        img = img.convert('RGB')
        images.append(img)
        # images.append(img_path)
    im1 = images[0]
    images = images[1:]
    im1.save(pdf_path)
    # with open(pdf_path, "w") as f: 
    #     f.write(img2pdf.convert(images.encode('UTF-8')))

def map_counter_id(counter, id, json_file_path, csv_file_path):
    # json_file_path = "counter_id_mapping.json"
    try:
        with open(json_file_path, 'r') as json_file:
            existing_data = json.load(json_file)
    except FileNotFoundError:
        existing_data = {}
    existing_data[counter] = id
    with open(json_file_path, 'w') as json_file:
        json.dump(existing_data, json_file)
    # csv_file_path = "counter_id_mapping.csv"
    with open(csv_file_path, 'a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([counter, id])

HEIGHT = 1683
WIDTH = 1190    
# PDF_DIR = "/home/suraj/Desktop/KCDH2/websiteForKCDH/test_pdf_dir"
# JSON_DIR = "/home/suraj/Desktop/KCDH2/websiteForKCDH/test_json_dir"
# PDF_DIR = "/home/sarveshdeshpande/Documents/IJCAI_Data/pdf"
# JSON_DIR = "/home/sarveshdeshpande/Documents/IJCAI_Data/json"


# JSON_FILE_PATH = "/home/sarveshdeshpande/Documents/Anonymization/Logs/mappings.json"
# CSV_FILE_PATH = "/home/sarveshdeshpande/Documents/Anonymization/Logs/mappings.csv"
def main(pdf_path, json_path, new_pdf_path):
        # idx = 0
        # pdf_path = os.path.join(PDF_DIR,fname)
    # new_pdf_path = os.path.join(NEW_PDFS_DIR, fname)
    # json_path = os.path.join(JSON_DIR, fname.replace('.pdf', '.json'))
    Labels=['PATIENT_ID','PATIENT_NAME','LOCATION', 'OTHERS']
    print(new_pdf_path)
    print(pdf_path)
    print(json_path)
    # width, height = pdf_to_image(pdf_path, IMG_DIR)
    width,height = 600,840
    demo, preds_list = get_model_preds(pdf_path, json_path)
    word = demo[0][0]
    label = preds_list[0]
    bbox_viz = demo[2][0]
    bbox_viz = denormalize(bbox_viz, WIDTH, HEIGHT)
    bbox_viz = translate(bbox_viz, WIDTH,HEIGHT, width, height)
    all_data = list(zip(word, label, bbox_viz))
    ocr_data = pd.DataFrame(all_data, columns=['word', 'label', 'bbox_viz'])
    data = list(ocr_data.itertuples(index=False, name=None))
    to_mask = ['PATIENT_ID', 'PATIENT_NAME', 'LOCATION']
    bboxes = []
    labels = []
    words = []
    patientId = ""
    for word, label, bbox_viz in data:
        if label in to_mask:
            # print(word, bbox_viz, label)
            # labels.append(Labels[label])
            words.append(word)
            labels.append(label)
            bboxes.append(bbox_viz)
            if label=="PATIENT_ID":
                patientId = word
    anonymize(pdf_path, new_pdf_path, bboxes, labels, words)

    # map_counter_id(idx, patientId, JSON_FILE_PATH, CSV_FILE_PATH)
    # img_to_pdf(NEW_IMG_DIR, new_pdf_path, idx)
    
    # idx+=1
    
    

#JSON_PATH='/home/suraj/Desktop/KCDH2/websiteForKCDH/test_json_dir/KH1000001526 jan.json'
#PDF_PATH='/home/suraj/Desktop/KCDH2/websiteForKCDH/test_pdf_dir/KH1000001526 jan.pdf'
#NEW_PDFS_PATH='/home/suraj/Desktop/KCDH2/websiteForKCDH/masked2.pdf'
#main(PDF_PATH, JSON_PATH, NEW_PDFS_PATH)