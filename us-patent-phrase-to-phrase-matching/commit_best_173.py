# %% [markdown]
# ## Analysis and using models from three notebooks
# 
# **1.** Deberta v3 large (0.8392)
# > [Inference BERT for usPatents](https://www.kaggle.com/code/leehann/inference-bert-for-uspatents)
# 
# **2.** Deberta v3 large (0.8338)
# > [PPPM / Deberta-v3-large baseline [inference]](https://www.kaggle.com/code/yasufuminakama/pppm-deberta-v3-large-baseline-inference)
# 
# **3.** Roberta-large (0.8143)
# > [PatentPhrase RoBERTa Inference](https://www.kaggle.com/code/santhoshkumarv/patentphrase-roberta-inference-lb-0-814)
# 
# #### Please upvote the original notebooks!
# 
# ## UPD: I have an error in my code (Version 1)!
# 
# Method merge in model 1 shuffled the dataframe.
# 
# ```
# test = test.merge(titles, left_on='context', right_on='code')
# ```
# 
# So I reseted index, merged, sorted and drop index.
# 
# ```
# test.reset_index(inplace=True)
# test = test.merge(titles, left_on='context', right_on='code')
# test.sort_values(by='index', inplace=True)
# test.drop(columns='index', inplace=True)
# ```

# %% [markdown]
# Start 8354?

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:32:25.249689Z","iopub.execute_input":"2022-06-20T16:32:25.250457Z","iopub.status.idle":"2022-06-20T16:32:25.280902Z","shell.execute_reply.started":"2022-06-20T16:32:25.250363Z","shell.execute_reply":"2022-06-20T16:32:25.280261Z"}}
# ====================================================
# Directory settings
# ====================================================
import os

INPUT_DIR = '../input/us-patent-phrase-to-phrase-matching/'
OUTPUT_DIR = './'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:32:25.282399Z","iopub.execute_input":"2022-06-20T16:32:25.282721Z","iopub.status.idle":"2022-06-20T16:32:25.289035Z","shell.execute_reply.started":"2022-06-20T16:32:25.282685Z","shell.execute_reply":"2022-06-20T16:32:25.287762Z"}}
# ====================================================
# CFG
# ====================================================
class CFG8354:
    num_workers=2
#     path="../input/us-patent-attention-largev3-model/large51/"
    path="../input/fork-of-us-patent-wtrain-model/wtrain/"
    config_path=path+'config.pth'
    model="microsoft/deberta-v3-large"
    batch_size=16#32
    fc_dropout=0.2
    target_size=1
    max_len=133
    seed=91
    n_fold=5
    trn_fold=[0, 1, 2, 3, 4]

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:32:25.290459Z","iopub.execute_input":"2022-06-20T16:32:25.291337Z","iopub.status.idle":"2022-06-20T16:32:58.759074Z","shell.execute_reply.started":"2022-06-20T16:32:25.291299Z","shell.execute_reply":"2022-06-20T16:32:58.758317Z"}}
# ====================================================
# Library
# ====================================================
import os
import gc
import re
import ast
import sys
import copy
import json
import time
import math
import shutil
import string
import pickle
import random
import joblib
import itertools
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import scipy as sp
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

import torch
print(f"torch.__version__: {torch.__version__}")
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset

os.system('pip uninstall -y transformers')
os.system('pip uninstall -y tokenizers')
os.system('python -m pip install --no-index --find-links=../input/pppm-pip-wheels-dataset transformers')
os.system('python -m pip install --no-index --find-links=../input/pppm-pip-wheels-dataset tokenizers')
import tokenizers
import transformers
print(f"tokenizers.__version__: {tokenizers.__version__}")
print(f"transformers.__version__: {transformers.__version__}")
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
%env TOKENIZERS_PARALLELISM=true

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:32:58.761412Z","iopub.execute_input":"2022-06-20T16:32:58.761668Z","iopub.status.idle":"2022-06-20T16:32:58.774646Z","shell.execute_reply.started":"2022-06-20T16:32:58.761633Z","shell.execute_reply":"2022-06-20T16:32:58.773840Z"}}
# ====================================================
# Utils
# ====================================================
def get_score(y_true, y_pred):
    score = sp.stats.pearsonr(y_true, y_pred)[0]
    return score


def get_logger(filename=OUTPUT_DIR+'train'):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

LOGGER = get_logger()

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed=CFG8354.seed)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:32:58.776163Z","iopub.execute_input":"2022-06-20T16:32:58.776621Z","iopub.status.idle":"2022-06-20T16:32:58.815470Z","shell.execute_reply.started":"2022-06-20T16:32:58.776587Z","shell.execute_reply":"2022-06-20T16:32:58.814619Z"}}
# ====================================================
# Data Loading
# ====================================================
test = pd.read_csv(INPUT_DIR+'test.csv')
submission = pd.read_csv(INPUT_DIR+'sample_submission.csv')
print(f"test.shape: {test.shape}")
print(f"submission.shape: {submission.shape}")
display(test.head())
display(submission.head())

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:32:58.816929Z","iopub.execute_input":"2022-06-20T16:32:58.817355Z","iopub.status.idle":"2022-06-20T16:32:58.845449Z","shell.execute_reply.started":"2022-06-20T16:32:58.817315Z","shell.execute_reply":"2022-06-20T16:32:58.844676Z"}}
# ====================================================
# CPC Data
# ====================================================
# cpc_texts = torch.load(CFG.path+"cpc_texts.pth")
cpc_texts = torch.load("../input/folds-dump-the-two-paths-fix/cpc_texts_fixed.pth")


test['context_text'] = test['context'].map(cpc_texts)
display(test.head())

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:32:58.846552Z","iopub.execute_input":"2022-06-20T16:32:58.847029Z","iopub.status.idle":"2022-06-20T16:32:58.861544Z","shell.execute_reply.started":"2022-06-20T16:32:58.846992Z","shell.execute_reply":"2022-06-20T16:32:58.860728Z"}}
# test['text'] = test['anchor'] + ' ' + test['context_text'].apply(str.lower) + ' '  + test['target'].apply(str.lower)
test['text'] = test['anchor'] + '[SEP]' + test['target'] + '[SEP]'  + test['context_text']
display(test.head())

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:32:58.863029Z","iopub.execute_input":"2022-06-20T16:32:58.863291Z","iopub.status.idle":"2022-06-20T16:32:59.619246Z","shell.execute_reply.started":"2022-06-20T16:32:58.863256Z","shell.execute_reply":"2022-06-20T16:32:59.618425Z"}}
# ====================================================
# tokenizer
# ====================================================
# CFG.tokenizer = AutoTokenizer.from_pretrained(CFG.path+'tokenizer/')
CFG8354.tokenizer = AutoTokenizer.from_pretrained('../input/fork-of-us-patent-wtrain-model/wtrain/tokenizer/')

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:32:59.620566Z","iopub.execute_input":"2022-06-20T16:32:59.621064Z","iopub.status.idle":"2022-06-20T16:32:59.629664Z","shell.execute_reply.started":"2022-06-20T16:32:59.621026Z","shell.execute_reply":"2022-06-20T16:32:59.628570Z"}}
# ====================================================
# Dataset
# ====================================================
def prepare_input(cfg, text):
    inputs = cfg.tokenizer(text,
                           add_special_tokens=True,
                           max_length=cfg.max_len,
                           padding="max_length",
                           return_offsets_mapping=False)
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


class TestDataset8354(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.texts = df['text'].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, self.texts[item])
        return inputs

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:32:59.633460Z","iopub.execute_input":"2022-06-20T16:32:59.633674Z","iopub.status.idle":"2022-06-20T16:32:59.650769Z","shell.execute_reply.started":"2022-06-20T16:32:59.633643Z","shell.execute_reply":"2022-06-20T16:32:59.649837Z"}}
# ====================================================
# Model
# ====================================================
class CustomModel8354(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)
        self.fc_dropout = nn.Dropout(cfg.fc_dropout)
        self.fc = nn.Linear(self.config.hidden_size, self.cfg.target_size)
        self._init_weights(self.fc)
        self.attention = nn.Sequential(
            nn.Linear(self.config.hidden_size, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )
        self._init_weights(self.attention)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        # feature = torch.mean(last_hidden_states, 1)
        weights = self.attention(last_hidden_states)
        feature = torch.sum(weights * last_hidden_states, dim=1)
        return feature

    def forward(self, inputs):
        feature = self.feature(inputs)
        output = self.fc(self.fc_dropout(feature))
        return output

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:32:59.652371Z","iopub.execute_input":"2022-06-20T16:32:59.652750Z","iopub.status.idle":"2022-06-20T16:32:59.664035Z","shell.execute_reply.started":"2022-06-20T16:32:59.652708Z","shell.execute_reply":"2022-06-20T16:32:59.663132Z"}}
# ====================================================
# inference
# ====================================================
def inference_fn8354(test_loader, model, device):
    preds = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader))
    for inputs in tk0:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        preds.append(y_preds.sigmoid().to('cpu').numpy())
    predictions = np.concatenate(preds)
    return predictions

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:32:59.665570Z","iopub.execute_input":"2022-06-20T16:32:59.665973Z","iopub.status.idle":"2022-06-20T16:36:11.027800Z","shell.execute_reply.started":"2022-06-20T16:32:59.665936Z","shell.execute_reply":"2022-06-20T16:36:11.018723Z"}}
test_dataset = TestDataset8354(CFG8354, test)
test_loader = DataLoader(test_dataset,
                         batch_size=CFG8354.batch_size,
                         shuffle=False,
                         num_workers=CFG8354.num_workers, pin_memory=True, drop_last=False)
predictions8354 = []
for fold in CFG8354.trn_fold:
    model = CustomModel8354(CFG8354, config_path=CFG8354.config_path, pretrained=False)
    state = torch.load(CFG8354.path+f"-content-drive-MyDrive-us-patent-deberta-v3-large-deberta-v3-large_fold{fold}_best.pth",                       
                       map_location=torch.device('cpu'))

    model.load_state_dict(state['model'])
    prediction = inference_fn8354(test_loader, model, device)
    predictions8354.append(prediction)
    del model, state, prediction; gc.collect()
    torch.cuda.empty_cache()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:36:11.029727Z","iopub.execute_input":"2022-06-20T16:36:11.030085Z","iopub.status.idle":"2022-06-20T16:36:11.218360Z","shell.execute_reply.started":"2022-06-20T16:36:11.030032Z","shell.execute_reply":"2022-06-20T16:36:11.212369Z"}}
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
cm = sns.light_palette('green', as_cmap=True)


def upd_outputs(data, is_trim=False, is_minmax=False, is_reshape=False):
    min_max_scaler = MinMaxScaler()
    
    if is_trim == True:
        data = np.where(data <=0, 0, data)
        data = np.where(data >=1, 1, data)

    if is_minmax ==True:
        data = min_max_scaler.fit_transform(data)
    
    if is_reshape == True:
        data = data.reshape(-1)
        
    return data

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:36:11.219715Z","iopub.execute_input":"2022-06-20T16:36:11.219986Z","iopub.status.idle":"2022-06-20T16:36:11.450454Z","shell.execute_reply.started":"2022-06-20T16:36:11.219953Z","shell.execute_reply":"2022-06-20T16:36:11.449669Z"}}
predictions8354_1 = [upd_outputs(x, is_minmax=True, is_reshape=True) for x in predictions8354]

predictions8354_1 = pd.DataFrame(predictions8354_1).T

predictions8354_1.head(10).style.background_gradient(cmap=cm, axis=1)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:36:11.454724Z","iopub.execute_input":"2022-06-20T16:36:11.457161Z","iopub.status.idle":"2022-06-20T16:36:11.465392Z","shell.execute_reply.started":"2022-06-20T16:36:11.457108Z","shell.execute_reply":"2022-06-20T16:36:11.464421Z"}}
class CFGBert:
    num_workers=2
#     path="../input/us-patent-attention-largev3-model/large51/"
    path="../input/us-patent-bertpatent-model/bertpatent/"
    config_path=path+'config.pth'
    model="../input/us-patent-bertpatent-model/bertpatent/"
    batch_size=16#32
    fc_dropout=0.2
    target_size=1
    max_len=133
    seed=102
    n_fold=5
    trn_fold=[0, 1, 2, 3, 4]

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:36:11.466965Z","iopub.execute_input":"2022-06-20T16:36:11.467547Z","iopub.status.idle":"2022-06-20T16:36:11.479020Z","shell.execute_reply.started":"2022-06-20T16:36:11.467505Z","shell.execute_reply":"2022-06-20T16:36:11.478098Z"}}
seed_everything(seed=CFGBert.seed)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:36:11.480502Z","iopub.execute_input":"2022-06-20T16:36:11.481761Z","iopub.status.idle":"2022-06-20T16:36:11.572435Z","shell.execute_reply.started":"2022-06-20T16:36:11.481720Z","shell.execute_reply":"2022-06-20T16:36:11.571710Z"}}
CFGBert.tokenizer = AutoTokenizer.from_pretrained('../input/us-patent-bertpatent-model/bertpatent/tokenizer/')

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:36:11.576875Z","iopub.execute_input":"2022-06-20T16:36:11.577570Z","iopub.status.idle":"2022-06-20T16:38:57.264483Z","shell.execute_reply.started":"2022-06-20T16:36:11.577521Z","shell.execute_reply":"2022-06-20T16:38:57.256544Z"}}
test_dataset = TestDataset8354(CFGBert, test)
test_loader = DataLoader(test_dataset,
                         batch_size=CFGBert.batch_size,
                         shuffle=False,
                         num_workers=CFGBert.num_workers, pin_memory=True, drop_last=False)
predictionsBert = []
for fold in CFGBert.trn_fold:
    model = CustomModel8354(CFGBert, config_path=CFGBert.config_path, pretrained=False)
    state = torch.load(CFGBert.path+f"-content-drive-MyDrive-us-patent-bert-for-patents-bert-for-patents_fold{fold}_best.pth",                       
                       map_location=torch.device('cpu'))
    model.load_state_dict(state['model'])
    prediction = inference_fn8354(test_loader, model, device)
    predictionsBert.append(prediction)
    del model, state, prediction; gc.collect()
    torch.cuda.empty_cache()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:38:57.266209Z","iopub.execute_input":"2022-06-20T16:38:57.266499Z","iopub.status.idle":"2022-06-20T16:38:57.348207Z","shell.execute_reply.started":"2022-06-20T16:38:57.266464Z","shell.execute_reply":"2022-06-20T16:38:57.346600Z"}}
predictionsBert_1 = [upd_outputs(x, is_minmax=True, is_reshape=True) for x in predictionsBert]

predictionsBert_1 = pd.DataFrame(predictionsBert_1).T

predictionsBert_1.head(10).style.background_gradient(cmap=cm, axis=1)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:38:57.349354Z","iopub.execute_input":"2022-06-20T16:38:57.349633Z","iopub.status.idle":"2022-06-20T16:38:57.578943Z","shell.execute_reply.started":"2022-06-20T16:38:57.349598Z","shell.execute_reply":"2022-06-20T16:38:57.578038Z"}}
del test, test_dataset
gc.collect()

# %% [markdown]
# Add 847

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:38:57.581137Z","iopub.execute_input":"2022-06-20T16:38:57.581907Z","iopub.status.idle":"2022-06-20T16:39:50.649850Z","shell.execute_reply.started":"2022-06-20T16:38:57.581861Z","shell.execute_reply":"2022-06-20T16:39:50.649018Z"}}
# !pip install datasets
# !pip uninstall -y pyarrow==7.0.0
# !pip install pyarrow
# import pyarrow
# print(pyarrow.__version__)
import os
!pip install ../input/hashwhl1/xxhash-3.0.0-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
!pip install ../input/datasets1/datasets-1.18.4-py3-none-any.whl --no-deps
# print(datasets.__version__)

# os.system('python -m pip install --no-index --find-links=../input/datasets1/datasets-1.18.4-py3-none-any.whl datasets')

import datasets, transformers

from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.preprocessing import MinMaxScaler

import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import gc
import scipy as sp

import pandas as pd
import numpy as np
import torch


os.environ["WANDB_DISABLED"] = "true"

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:39:50.651323Z","iopub.execute_input":"2022-06-20T16:39:50.651573Z","iopub.status.idle":"2022-06-20T16:39:50.659215Z","shell.execute_reply.started":"2022-06-20T16:39:50.651537Z","shell.execute_reply":"2022-06-20T16:39:50.658300Z"}}
class CFG847:
    input_path = '../input/us-patent-phrase-to-phrase-matching/'
    model_path = ['../input/deberta-v3-5folds/',
                  '../input/bert-for-patent-5fold/', 
                  '../input/deberta-large-v1/',
                  '../input/xlm-roberta-large-5folds/',
                 ]
    model_num = 4
    num_fold = 5

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:39:50.660617Z","iopub.execute_input":"2022-06-20T16:39:50.660936Z","iopub.status.idle":"2022-06-20T16:39:51.606129Z","shell.execute_reply.started":"2022-06-20T16:39:50.660900Z","shell.execute_reply":"2022-06-20T16:39:51.605333Z"}}
titles = pd.read_csv('../input/cpc-codes/titles.csv')

test = pd.read_csv(f"{CFG847.input_path}test.csv")
test.reset_index(inplace=True)
test = test.merge(titles, left_on='context', right_on='code')
test.sort_values(by='index', inplace=True)
test.drop(columns='index', inplace=True)
test['input'] = test['title']+'[SEP]'+test['anchor']
test = test.drop(columns=["context", "code", "class", "subclass", "group", "main_group", "anchor", "title", "section"])

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:39:51.607746Z","iopub.execute_input":"2022-06-20T16:39:51.608052Z","iopub.status.idle":"2022-06-20T16:46:56.361452Z","shell.execute_reply.started":"2022-06-20T16:39:51.608002Z","shell.execute_reply":"2022-06-20T16:46:56.360584Z"}}
predictions847 = []
weights = [0.5, 0.3, 0.1, 0.1]
MMscaler = MinMaxScaler()

for i in range (CFG847.model_num):   
    tokenizer = AutoTokenizer.from_pretrained(f'{CFG847.model_path[i]}fold0')

    def process_test(unit):
            return {
            **tokenizer( unit['input'], unit['target'])
        }
    
    def process_valid(unit):
        return {
        **tokenizer( unit['input'], unit['target']),
        'label': unit['score']
    }
    
    test_ds = datasets.Dataset.from_pandas(test)
    test_ds = test_ds.map(process_test, remove_columns=['id', 'target', 'input', '__index_level_0__'])

    for fold in range(CFG847.num_fold):        
        model = AutoModelForSequenceClassification.from_pretrained(f'{CFG847.model_path[i]}fold{fold}', 
                                                                   num_labels=1)
        trainer = Trainer(
                model,
                tokenizer=tokenizer,
            )
        
        outputs = trainer.predict(test_ds)
        prediction = MMscaler.fit_transform(outputs.predictions.reshape(-1, 1)).reshape(-1) * (weights[i] / 5)
        predictions847.append(prediction)
        
        del model, prediction
        torch.cuda.empty_cache()
        gc.collect()
predictions847 = np.sum(predictions847, axis=0)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:46:56.362963Z","iopub.execute_input":"2022-06-20T16:46:56.363257Z","iopub.status.idle":"2022-06-20T16:46:56.570753Z","shell.execute_reply.started":"2022-06-20T16:46:56.363216Z","shell.execute_reply":"2022-06-20T16:46:56.569949Z"}}
del test, test_ds
gc.collect()

# %% [markdown]
# # 1. Import & Def & Set & Load

# %% [code] {"_kg_hide-input":true,"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:46:56.572462Z","iopub.execute_input":"2022-06-20T16:46:56.573694Z","iopub.status.idle":"2022-06-20T16:46:56.582448Z","shell.execute_reply.started":"2022-06-20T16:46:56.573649Z","shell.execute_reply":"2022-06-20T16:46:56.581527Z"}}
import os
import gc
import random

import scipy as sp
import numpy as np
import pandas as pd

import seaborn as sns

from sklearn.preprocessing import MinMaxScaler

from dataclasses import dataclass

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, AutoModel

import warnings 
warnings.filterwarnings('ignore')

# %% [code] {"_kg_hide-input":true,"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:46:56.584232Z","iopub.execute_input":"2022-06-20T16:46:56.584816Z","iopub.status.idle":"2022-06-20T16:46:56.597621Z","shell.execute_reply.started":"2022-06-20T16:46:56.584758Z","shell.execute_reply":"2022-06-20T16:46:56.596826Z"}}
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True    
    torch.backends.cudnn.benchmark = False

    
def inference_fn(test_loader, model, device, is_sigmoid=True):
    preds = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader))
    
    for inputs in tk0:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
            
        with torch.no_grad():
            output = model(inputs)
        
        if is_sigmoid == True:
            preds.append(output.sigmoid().to('cpu').numpy())
        else:
            preds.append(output.to('cpu').numpy())

    return np.concatenate(preds)    
    

def upd_outputs(data, is_trim=False, is_minmax=False, is_reshape=False):
    min_max_scaler = MinMaxScaler()
    
    if is_trim == True:
        data = np.where(data <=0, 0, data)
        data = np.where(data >=1, 1, data)

    if is_minmax ==True:
        data = min_max_scaler.fit_transform(data)
    
    if is_reshape == True:
        data = data.reshape(-1)
        
    return data

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:46:56.605248Z","iopub.execute_input":"2022-06-20T16:46:56.606045Z","iopub.status.idle":"2022-06-20T16:46:56.615833Z","shell.execute_reply.started":"2022-06-20T16:46:56.605996Z","shell.execute_reply":"2022-06-20T16:46:56.614840Z"}}
pd.set_option('display.precision', 4)
cm = sns.light_palette('green', as_cmap=True)
props_param = "color:white; font-weight:bold; background-color:green;"

CUSTOM_SEED = 42
CUSTOM_BATCH = 24
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:46:56.617560Z","iopub.execute_input":"2022-06-20T16:46:56.618272Z","iopub.status.idle":"2022-06-20T16:46:56.647606Z","shell.execute_reply.started":"2022-06-20T16:46:56.618226Z","shell.execute_reply":"2022-06-20T16:46:56.646759Z"}}
competition_dir = "../input/us-patent-phrase-to-phrase-matching/"

submission = pd.read_csv(competition_dir+'sample_submission.csv')
test_origin = pd.read_csv(competition_dir+'test.csv')
test_origin.head()

# %% [markdown]
# # 2. Extract predictions
# 
# ## 2.1 Deberta v3 large - 1

# %% [code] {"_kg_hide-input":true,"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:46:56.649181Z","iopub.execute_input":"2022-06-20T16:46:56.649450Z","iopub.status.idle":"2022-06-20T16:46:56.662368Z","shell.execute_reply.started":"2022-06-20T16:46:56.649419Z","shell.execute_reply":"2022-06-20T16:46:56.661457Z"}}
def prepare_input(cfg, text):
    inputs = cfg.tokenizer(text,
                           max_length=cfg.max_len,
                           padding="max_length",
                           truncation=True)
    
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
        
    return inputs

class TestDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg        
        self.text = df['text'].values
        
    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, self.text[item])
        
        return inputs
   
    
class CustomModel(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        
        config = AutoConfig.from_pretrained(model_path)
        config.num_labels = 1
        self.base = AutoModelForSequenceClassification.from_config(config=config)
        dim = config.hidden_size
        self.dropout = nn.Dropout(p=0)
        self.cls = nn.Linear(dim,1)
        
    def forward(self, inputs):
        output = self.base(**inputs)

        return output[0]

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:46:56.664281Z","iopub.execute_input":"2022-06-20T16:46:56.664885Z","iopub.status.idle":"2022-06-20T16:46:56.675609Z","shell.execute_reply.started":"2022-06-20T16:46:56.664843Z","shell.execute_reply":"2022-06-20T16:46:56.674831Z"}}
seed_everything(CUSTOM_SEED)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:46:56.678291Z","iopub.execute_input":"2022-06-20T16:46:56.678740Z","iopub.status.idle":"2022-06-20T16:46:57.478597Z","shell.execute_reply.started":"2022-06-20T16:46:56.678697Z","shell.execute_reply":"2022-06-20T16:46:57.477889Z"}}
class CFG:
    model_path='../input/deberta-v3-large/deberta-v3-large'
    batch_size=CUSTOM_BATCH
    num_workers=2
    max_len=130
    trn_fold=[0, 1, 2, 3]

CFG.tokenizer = AutoTokenizer.from_pretrained(CFG.model_path)

context_mapping = torch.load("../input/folds-dump-the-two-paths-fix/cpc_texts.pth")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:46:57.479770Z","iopub.execute_input":"2022-06-20T16:46:57.480370Z","iopub.status.idle":"2022-06-20T16:46:58.400159Z","shell.execute_reply.started":"2022-06-20T16:46:57.480330Z","shell.execute_reply":"2022-06-20T16:46:58.398412Z"}}
test = test_origin.copy()
titles = pd.read_csv('../input/cpc-codes/titles.csv')

test.reset_index(inplace=True)
test = test.merge(titles, left_on='context', right_on='code')
test.sort_values(by='index', inplace=True)
test.drop(columns='index', inplace=True)

test['context_text'] = test['context'].map(context_mapping)
test['text'] = test['anchor'] + '[SEP]' + test['target'] + '[SEP]'  + test['context_text']
test['text'] = test['text'].apply(str.lower)

test.head()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:46:58.401593Z","iopub.execute_input":"2022-06-20T16:46:58.402064Z","iopub.status.idle":"2022-06-20T16:48:54.747598Z","shell.execute_reply.started":"2022-06-20T16:46:58.402020Z","shell.execute_reply":"2022-06-20T16:48:54.746677Z"}}
deberta_predicts_1 = []

test_dataset = TestDataset(CFG, test)
test_dataloader = DataLoader(test_dataset,
                             batch_size=CFG.batch_size, shuffle=False,
                             num_workers=CFG.num_workers,
                             pin_memory=True, drop_last=False)

deberta_simple_path = "../input/us-patent-deberta-simple/microsoft_deberta-v3-large"

for fold in CFG.trn_fold:
    fold_path = f"{deberta_simple_path}_best{fold}.pth"
    
    model = CustomModel(CFG.model_path)    
    state = torch.load(fold_path, map_location=torch.device('cpu'))  # DEVICE
    model.load_state_dict(state['model'])
    
    prediction = inference_fn(test_dataloader, model, DEVICE, is_sigmoid=False)
    
    deberta_predicts_1.append(prediction)
    
    del model, state, prediction
    torch.cuda.empty_cache()
    gc.collect()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:48:54.749403Z","iopub.execute_input":"2022-06-20T16:48:54.749672Z","iopub.status.idle":"2022-06-20T16:48:54.786036Z","shell.execute_reply.started":"2022-06-20T16:48:54.749636Z","shell.execute_reply":"2022-06-20T16:48:54.785088Z"}}
# -------------- inference_fn([...], is_sigmoid=False)
deberta_predicts_1 = [upd_outputs(x, is_minmax=True, is_reshape=True) for x in deberta_predicts_1]
deberta_predicts_1 = pd.DataFrame(deberta_predicts_1).T

deberta_predicts_1.head(10).style.background_gradient(cmap=cm, axis=1)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:48:54.787816Z","iopub.execute_input":"2022-06-20T16:48:54.788121Z","iopub.status.idle":"2022-06-20T16:48:54.993304Z","shell.execute_reply.started":"2022-06-20T16:48:54.788078Z","shell.execute_reply":"2022-06-20T16:48:54.992498Z"}}
del test, test_dataset
gc.collect()

# %% [markdown]
# # 2.X LargeScore

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:48:54.994886Z","iopub.execute_input":"2022-06-20T16:48:54.995328Z","iopub.status.idle":"2022-06-20T16:48:55.004483Z","shell.execute_reply.started":"2022-06-20T16:48:54.995281Z","shell.execute_reply":"2022-06-20T16:48:55.003688Z"}}
import os
import gc
import pandas as pd
import numpy as np
import seaborn as sns


from tqdm import tqdm
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, AutoModel

from sklearn.preprocessing import MinMaxScaler

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:48:55.007561Z","iopub.execute_input":"2022-06-20T16:48:55.007843Z","iopub.status.idle":"2022-06-20T16:48:55.017417Z","shell.execute_reply.started":"2022-06-20T16:48:55.007810Z","shell.execute_reply":"2022-06-20T16:48:55.016467Z"}}
class CFG_DEB_SIMPLEScore:
    input_path = '../input/us-patent-phrase-to-phrase-matching/'
    model_path = '../input/deberta-v3-large/deberta-v3-large'
    batch_size = CUSTOM_BATCH#24
    num_workers = 2
    num_fold = 4#4
    max_input_length = 133#140
    max_len = 133
    trn_fold=[0, 1, 2, 3]

    seed = 86
    scheduler = 'cosine'  # ['linear', 'cosine']
    batch_scheduler = True
    num_cycles = 0.5  # 1.5
    num_warmup_steps = 0.1
    epochs = 4  # 5
    min_lr = 0.5e-6
    eps = 1e-6
    betas = (0.9, 0.999)
    learning_rate = 2e-6
    weight_decay = 0.01
    OUTPUT_DIR = './train_checkpoint'
    device='cuda'
    print_freq = 100

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:48:55.021036Z","iopub.execute_input":"2022-06-20T16:48:55.021277Z","iopub.status.idle":"2022-06-20T16:48:55.028322Z","shell.execute_reply.started":"2022-06-20T16:48:55.021249Z","shell.execute_reply":"2022-06-20T16:48:55.027251Z"}}
seed_everything(CUSTOM_SEED)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:48:55.031143Z","iopub.execute_input":"2022-06-20T16:48:55.031388Z","iopub.status.idle":"2022-06-20T16:48:55.043722Z","shell.execute_reply.started":"2022-06-20T16:48:55.031359Z","shell.execute_reply":"2022-06-20T16:48:55.043004Z"}}
class TestDatasetScore(Dataset):
    def __init__(self, df, tokenizer, max_input_length):
        self.text = df['text'].values.astype(str)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        
    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        inputs = self.text[item]
        
        inputs = self.tokenizer(inputs,
                    max_length=self.max_input_length,
                    padding='max_length',
                    truncation=True)
        
        return torch.as_tensor(inputs['input_ids'], dtype=torch.long), \
               torch.as_tensor(inputs['token_type_ids'], dtype=torch.long), \
               torch.as_tensor(inputs['attention_mask'], dtype=torch.long)
    
    
class Custom_Bert_SimpleScore(nn.Module):
    def __init__(self,model_path):
        super().__init__()
#         self.cfg = cfg
        config = AutoConfig.from_pretrained(model_path)
        config.num_labels = 1
        self.base = AutoModelForSequenceClassification.from_config(config=config)
        dim = config.hidden_size
        self.dropout = nn.Dropout(p=0)
        self.cls = nn.Linear(dim,1)
        

        
    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        base_output = self.base(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids
        )
        return base_output[0]

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:48:55.045134Z","iopub.execute_input":"2022-06-20T16:48:55.045723Z","iopub.status.idle":"2022-06-20T16:48:55.061845Z","shell.execute_reply.started":"2022-06-20T16:48:55.045679Z","shell.execute_reply":"2022-06-20T16:48:55.060947Z"}}
def valid_fnscore(valid_loader, model, device):
    model.eval()
    preds = []
    labels = []
    
    for step, batch in enumerate(valid_loader):
        input_ids, token_type_ids, attention_mask = [i.to(device) for i in batch]
    
        with torch.no_grad():
            y_preds = model(input_ids, attention_mask, token_type_ids)
        
        preds.append(y_preds.to('cpu').numpy())
    
    predictions = np.concatenate(preds)
    
    return predictions


min_max_scaler = MinMaxScaler()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:48:55.063516Z","iopub.execute_input":"2022-06-20T16:48:55.064106Z","iopub.status.idle":"2022-06-20T16:48:55.088370Z","shell.execute_reply.started":"2022-06-20T16:48:55.064063Z","shell.execute_reply":"2022-06-20T16:48:55.087653Z"}}
test_df = test_origin.copy()
cpc_texts = torch.load("../input/folds-dump-the-two-paths-fix/cpc_texts_fixed.pth")

test_df['context_text'] = test_df['context'].map(cpc_texts)
# test_df['text'] = test_df['anchor'] + ' ' + test_df['context_text'] + ' '  + test_df['target']
test_df['text'] = '[CLS]' + test_df['context_text'] + '[SEP]' + test_df['anchor'] + '[SEP]' + test_df['target']

test_df['text'] = test_df['text'].apply(str.lower)

test_df.head()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:48:55.089564Z","iopub.execute_input":"2022-06-20T16:48:55.090216Z","iopub.status.idle":"2022-06-20T16:48:55.901030Z","shell.execute_reply.started":"2022-06-20T16:48:55.090174Z","shell.execute_reply":"2022-06-20T16:48:55.900204Z"}}
CFG_DEB_SIMPLEScore.tokenizer = AutoTokenizer.from_pretrained(CFG_DEB_SIMPLEScore.model_path)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:48:55.902659Z","iopub.execute_input":"2022-06-20T16:48:55.903076Z","iopub.status.idle":"2022-06-20T16:51:20.753668Z","shell.execute_reply.started":"2022-06-20T16:48:55.903035Z","shell.execute_reply":"2022-06-20T16:51:20.752838Z"}}
# ====================================================
# Define max_len
# ====================================================
from tqdm.auto import tqdm
MMscaler = MinMaxScaler()

lengths_dict = {}

lengths = []
tk0 = tqdm(cpc_texts.values(), total=len(cpc_texts))
for text in tk0:
    length = len(CFG_DEB_SIMPLEScore.tokenizer(text, add_special_tokens=False)['input_ids'])
    lengths.append(length)
lengths_dict['context_text'] = lengths

for text_col in ['anchor', 'target']:
    lengths = []
    tk0 = tqdm(test_df[text_col].fillna("").values, total=len(test_df))
    for text in tk0:
        length = len(CFG_DEB_SIMPLEScore.tokenizer(text, add_special_tokens=False)['input_ids'])
        lengths.append(length)
    lengths_dict[text_col] = lengths
    

CFG_DEB_SIMPLEScore.max_len = max(lengths_dict['anchor']) + max(lengths_dict['target'])\
                + max(lengths_dict['context_text']) + 4 # CLS + SEP + SEP + SEP
print(f"max_len: {CFG_DEB_SIMPLEScore.max_len}")   



predictionsscore = []

te_dataset = TestDatasetScore(test_df, CFG_DEB_SIMPLEScore.tokenizer, CFG_DEB_SIMPLEScore.max_input_length)


te_dataloader = DataLoader(te_dataset,
                              batch_size=CFG_DEB_SIMPLEScore.batch_size, shuffle=False,
                              num_workers=CFG_DEB_SIMPLEScore.num_workers,
                              pin_memory=True, drop_last=False)


deberta_simple_path = "../input/us-patent-score-largev3-model/large6score/_content_drive_MyDrive_us-patent_deberta-v3-large_deberta-v3-large"




for fold in CFG_DEB_SIMPLEScore.trn_fold:
    fold_path = f"{deberta_simple_path}_best{fold}.pth"
#     print("fold_path",fold_path)
    model = Custom_Bert_SimpleScore(CFG_DEB_SIMPLEScore.model_path)
    model.load_state_dict(torch.load(fold_path,  map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))['model'])
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    prediction = valid_fnscore(te_dataloader, model, 'cuda' if torch.cuda.is_available() else 'cpu')
#     print("prediction",prediction)
 
    predictionsscore.append(prediction)
    del model, prediction
    torch.cuda.empty_cache()
    gc.collect()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:51:20.755149Z","iopub.execute_input":"2022-06-20T16:51:20.755646Z","iopub.status.idle":"2022-06-20T16:51:20.791087Z","shell.execute_reply.started":"2022-06-20T16:51:20.755601Z","shell.execute_reply":"2022-06-20T16:51:20.790080Z"}}
pd.set_option('display.precision', 4)
cm = sns.light_palette('green', as_cmap=True)
props_param = "color:white; font-weight:bold; background-color:green;"

# -------------- inference_fn([...], is_sigmoid=False)
predictionsscore_1 = [upd_outputs(x, is_minmax=True, is_reshape=True) for x in predictionsscore]

predictionsscore_1 = pd.DataFrame(predictionsscore_1).T

predictionsscore_1.head(10).style.background_gradient(cmap=cm, axis=1)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:51:20.792565Z","iopub.execute_input":"2022-06-20T16:51:20.793421Z","iopub.status.idle":"2022-06-20T16:51:21.007686Z","shell.execute_reply.started":"2022-06-20T16:51:20.793372Z","shell.execute_reply":"2022-06-20T16:51:21.006876Z"}}
del test_df, te_dataset
gc.collect()

# %% [markdown]
# #Add Electra

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:51:21.009098Z","iopub.execute_input":"2022-06-20T16:51:21.009590Z","iopub.status.idle":"2022-06-20T16:51:21.019647Z","shell.execute_reply.started":"2022-06-20T16:51:21.009546Z","shell.execute_reply":"2022-06-20T16:51:21.018706Z"}}
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedGroupKFold
import shutil
import time
import gc
import random
import math
import torch
from torch.utils.data import DataLoader, Dataset
import transformers
from transformers import TrainingArguments, Trainer, DataCollatorForWholeWordMask
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, AutoModel
from torch import nn
from torch.optim import Adam, SGD, AdamW
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
os.environ["WANDB_DISABLED"] = "true"

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:51:21.021155Z","iopub.execute_input":"2022-06-20T16:51:21.021544Z","iopub.status.idle":"2022-06-20T16:51:21.031142Z","shell.execute_reply.started":"2022-06-20T16:51:21.021504Z","shell.execute_reply":"2022-06-20T16:51:21.030220Z"}}
class CFGElect:
    input_path = '../input/us-patent-phrase-to-phrase-matching/'
    model_path = '../input/electra/large-discriminator'

    scheduler = 'cosine'  # ['linear', 'cosine']
    batch_scheduler = True
    num_cycles = 0.5  # 1.5
    num_warmup_steps = 0.1
    max_input_length = 140
    epochs = 5  # 5
    min_lr = 0.5e-6
    eps = 1e-6
    betas = (0.9, 0.999)
    learning_rate = 5e-6
    weight_decay = 0.01
    num_fold = 5
    trn_fold=[0,1,2,3,4]
    batch_size = 32#100
    seed = 86#19800102
    OUTPUT_DIR = './train_checkpoint'
    num_workers = 4#2
    device='cuda'
    print_freq = 100
    fc_dropout=0.2

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:51:21.032586Z","iopub.execute_input":"2022-06-20T16:51:21.032819Z","iopub.status.idle":"2022-06-20T16:51:21.135100Z","shell.execute_reply.started":"2022-06-20T16:51:21.032772Z","shell.execute_reply":"2022-06-20T16:51:21.134224Z"}}
from transformers import ElectraTokenizer, ElectraForSequenceClassification,AdamW #Huggingface transformer algorithms and pretrain weights.

CFGElect.tokenizer = ElectraTokenizer.from_pretrained(CFGElect.model_path)

context_mapping = torch.load("../input/folds-dump-the-two-paths-fix/cpc_texts.pth")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:51:21.136981Z","iopub.execute_input":"2022-06-20T16:51:21.137270Z","iopub.status.idle":"2022-06-20T16:51:21.151646Z","shell.execute_reply.started":"2022-06-20T16:51:21.137231Z","shell.execute_reply":"2022-06-20T16:51:21.150442Z"}}
class TestDatasetElect(Dataset):
    def __init__(self, df, tokenizer, max_input_length):
        self.text = df['text'].values.astype(str)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        
    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        inputs = self.text[item]
        
        inputs = self.tokenizer(inputs,
                    max_length=self.max_input_length,
                    padding='max_length',
                    truncation=True)
        
        return torch.as_tensor(inputs['input_ids'], dtype=torch.long), \
               torch.as_tensor(inputs['token_type_ids'], dtype=torch.long), \
               torch.as_tensor(inputs['attention_mask'], dtype=torch.long)
    
    
class Custom_Bert_SimpleElect(nn.Module):
    def __init__(self,model_path):
        super().__init__()
#         self.cfg = cfg
        config = AutoConfig.from_pretrained(model_path)
        config.num_labels = 1
        self.base = AutoModelForSequenceClassification.from_config(config=config)
        dim = config.hidden_size
        self.dropout = nn.Dropout(p=0)
        self.cls = nn.Linear(dim,1)
        

        
    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        base_output = self.base(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids
        )
        return base_output[0]
def valid_fnscoreElect(valid_loader, model, device):
    model.eval()
    model.to(device)
    
    preds = []
    labels = []
    
    for step, batch in enumerate(valid_loader):
        input_ids, token_type_ids, attention_mask = [i.to(device) for i in batch]
    
        with torch.no_grad():
            y_preds = model(input_ids, attention_mask, token_type_ids)
        
        preds.append(y_preds.to('cpu').numpy())
    
    predictions = np.concatenate(preds)
    
    return predictions


min_max_scaler = MinMaxScaler()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:51:21.153842Z","iopub.execute_input":"2022-06-20T16:51:21.154215Z","iopub.status.idle":"2022-06-20T16:51:21.955540Z","shell.execute_reply.started":"2022-06-20T16:51:21.154172Z","shell.execute_reply":"2022-06-20T16:51:21.954767Z"}}
test = test_origin.copy()
titles = pd.read_csv('../input/cpc-codes/titles.csv')
test['context_text'] = test['context'].map(context_mapping)
test['text'] = '[CLS]' + test['context_text'] + '[SEP]' + test['anchor'] + '[SEP]' + test['target']
test['text'] = test['text'].apply(str.lower)
test.head()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:51:21.957139Z","iopub.execute_input":"2022-06-20T16:51:21.957442Z","iopub.status.idle":"2022-06-20T16:53:46.744261Z","shell.execute_reply.started":"2022-06-20T16:51:21.957393Z","shell.execute_reply":"2022-06-20T16:53:46.743209Z"}}
electra_predicts = []

test_dataset = TestDatasetElect(test, CFGElect.tokenizer, CFGElect.max_input_length)

test_dataloader = DataLoader(test_dataset,
                             batch_size=CFGElect.batch_size, shuffle=False,
                             num_workers=CFGElect.num_workers,
                             pin_memory=True, drop_last=False)

deberta_simple_path = "../input/fork-of-us-patent-electra-model1/electra/_content_drive_MyDrive_us-patent_google-electra-large-discriminator"

for fold in CFGElect.trn_fold:
    fold_path = f"{deberta_simple_path}_best{fold}.pth"
    
    model = Custom_Bert_SimpleElect(CFGElect.model_path)    
    state = torch.load(fold_path, map_location=torch.device('cpu'))  # DEVICE
    
    model.load_state_dict(state['model'])
    
    prediction = valid_fnscoreElect(test_dataloader, model, 'cuda' if torch.cuda.is_available() else 'cpu')
    
#     prediction = valid_fnscoreElect(test_dataloader, model, 'cuda')
    
    electra_predicts.append(prediction)
    
    del model, state, prediction
    torch.cuda.empty_cache()
    gc.collect()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:53:46.746376Z","iopub.execute_input":"2022-06-20T16:53:46.747549Z","iopub.status.idle":"2022-06-20T16:53:46.784554Z","shell.execute_reply.started":"2022-06-20T16:53:46.747466Z","shell.execute_reply":"2022-06-20T16:53:46.783396Z"}}
electra_predicts_1 = [upd_outputs(x, is_minmax=True, is_reshape=True) for x in electra_predicts]

electra_predicts_1 = pd.DataFrame(electra_predicts_1).T

electra_predicts_1.head(10).style.background_gradient(cmap=cm, axis=1)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:53:46.786079Z","iopub.execute_input":"2022-06-20T16:53:46.786576Z","iopub.status.idle":"2022-06-20T16:53:47.012973Z","shell.execute_reply.started":"2022-06-20T16:53:46.786534Z","shell.execute_reply":"2022-06-20T16:53:47.011904Z"}}
del test, test_dataset
gc.collect()

# %% [markdown]
# # 3. Comparison / Ensemble

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:53:47.014446Z","iopub.execute_input":"2022-06-20T16:53:47.014892Z","iopub.status.idle":"2022-06-20T16:53:47.163449Z","shell.execute_reply.started":"2022-06-20T16:53:47.014852Z","shell.execute_reply":"2022-06-20T16:53:47.162626Z"}}
all_predictions = pd.concat(
#     [deberta_predicts_1, deberta_predicts_2, roberta_predicts, predictions8369_1, predictionsscore_1, electra_predicts_1,predictions8354_1],
    [deberta_predicts_1,predictionsscore_1,electra_predicts_1,predictions8354_1,predictionsBert_1],
    
#     keys=['deberta 1', 'deberta 2', 'roberta', 'predictions8369_1','predictionsscore_1', 'electra_predicts_1','predictions8354_1'],
    keys=['deberta 1','predictionsscore_1','electra_predicts_1','predictions8354_1','predictionsBert_1'],
    axis=1
)

all_predictions.head(10) \
    .assign(mean=lambda x: x.mean(axis=1)) \
        .style.background_gradient(cmap=cm, axis=1)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:53:47.165073Z","iopub.execute_input":"2022-06-20T16:53:47.165577Z","iopub.status.idle":"2022-06-20T16:53:47.196402Z","shell.execute_reply.started":"2022-06-20T16:53:47.165538Z","shell.execute_reply":"2022-06-20T16:53:47.195284Z"}}
all_mean = pd.DataFrame({
    'deberta 1': deberta_predicts_1.mean(axis=1),
#     'deberta 2': deberta_predicts_2.mean(axis=1),
#     'roberta': roberta_predicts.mean(axis=1),
#     'predictions8369_1' : predictions8369_1.mean(axis=1),
    'predictionsscore_1' : predictionsscore_1.mean(axis=1),
    'electra_predicts_1' : electra_predicts_1.mean(axis=1),  
    'predictions8354_1' : predictions8354_1.mean(axis=1), 
    'predictionsBert_1' : predictionsBert_1.mean(axis=1) 
})

all_mean.head(10) \
    .assign(mean=lambda x: x.mean(axis=1)) \
        .style.highlight_max(axis=1, props=props_param)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:53:47.198997Z","iopub.execute_input":"2022-06-20T16:53:47.199322Z","iopub.status.idle":"2022-06-20T16:53:47.214032Z","shell.execute_reply.started":"2022-06-20T16:53:47.199280Z","shell.execute_reply":"2022-06-20T16:53:47.212839Z"}}
# === N1 ===
# weights_ = [0.33, 0.33, 0.33]
# final_predictions = all_mean.mul(weights_).sum(axis=1)

# === N2 ===
# final_predictions = all_mean.median(axis=1)
# final_predictions = all_mean.mean(axis=1)
final_predictions = all_mean.mean(axis=1) * 0.9 + predictions847 * 0.1


# === N3 ===
# final_predictions = all_predictions.mean(axis=1)

# === N4 ===
# combs = pd.DataFrame({
#     'deberta_1': deberta_predicts_1.mean(axis=1),
#     'deb_2+rob': (deberta_predicts_2.mean(axis=1) * 0.666) \
#                     + (roberta_predicts.mean(axis=1) * 0.333)
# })
# display(combs.head())
# final_predictions = combs.median(axis=1)
# final_predictions = combs.mean(axis=1)

final_predictions.head()

# %% [markdown]
# # 4. Submission

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:53:47.216028Z","iopub.execute_input":"2022-06-20T16:53:47.216496Z","iopub.status.idle":"2022-06-20T16:53:47.234845Z","shell.execute_reply.started":"2022-06-20T16:53:47.216378Z","shell.execute_reply":"2022-06-20T16:53:47.234055Z"}}
submission = pd.DataFrame({
    'id': test_origin['id'],
    'score': final_predictions,
})

submission.head(14)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:53:47.238948Z","iopub.execute_input":"2022-06-20T16:53:47.239729Z","iopub.status.idle":"2022-06-20T16:53:47.253974Z","shell.execute_reply.started":"2022-06-20T16:53:47.239695Z","shell.execute_reply":"2022-06-20T16:53:47.252875Z"}}
def _upd_score_between(data, thresholds, value):
    """\o/"""
    mask_th = data.between(*thresholds, inclusive='both')
    data[mask_th] = value


def upd_score(data, th_dict=None):
    """\o/"""
    if isinstance(data, pd.Series):
        result = data.copy()
    else:
        return data

    if not th_dict:        
        th_dict = {
            '0': 0.02,
            '.25': (0.24, 0.26),
            '.50': (0.49, 0.51),
            '.75': (0.74, 0.76),
            '1': 0.98
        }

    if isinstance(th_dict, dict):    
        th0 = th_dict.get('0')
        th25 = th_dict.get('.25')
        th50 = th_dict.get('.50')
        th75 = th_dict.get('.75')
        th100 = th_dict.get('1')
    else:
        return data
    
    if th0:
        if isinstance(th0, float):
            th0 = (result.min(), th0)
        
        if isinstance(th0, tuple):
            _upd_score_between(result, th0, 0)
    
    if th25 and isinstance(th25, tuple):
        _upd_score_between(result, th25, 0.25)

    if th50 and isinstance(th50, tuple):
        _upd_score_between(result, th50, 0.50)
            
    if th75 and isinstance(th75, tuple):
        _upd_score_between(result, th75, 0.75)
            
    if th100:
        if isinstance(th100, float):
            th100 = (th100, result.max())
        
        if isinstance(th100, tuple):
            _upd_score_between(result, th100, 1)

    return result

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:53:47.255811Z","iopub.execute_input":"2022-06-20T16:53:47.256506Z","iopub.status.idle":"2022-06-20T16:53:47.287912Z","shell.execute_reply.started":"2022-06-20T16:53:47.256461Z","shell.execute_reply":"2022-06-20T16:53:47.286842Z"}}
thresholds_dict = {
    '0': 0.02,
    '.25': (0.24, 0.26),
    '.50': (0.49, 0.51),
    '.75': (0.74, 0.76),
    '1': 0.98
}

submission['score'] = upd_score(submission['score'], thresholds_dict)

submission.head(14)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-06-20T16:53:47.289647Z","iopub.execute_input":"2022-06-20T16:53:47.290499Z","iopub.status.idle":"2022-06-20T16:53:47.302102Z","shell.execute_reply.started":"2022-06-20T16:53:47.290452Z","shell.execute_reply":"2022-06-20T16:53:47.300764Z"}}
submission.to_csv('submission.csv', index=False)

# %% [code] {"jupyter":{"outputs_hidden":false}}
