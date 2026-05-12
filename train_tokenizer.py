import requests
import re
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tokenizers.processors import TemplateProcessing
import sentencepiece as spm
import os

# 1. Fetch dataset from GitHub
print('Fetching Tiny Shakespeare...')
url = 'https://raw.githubusercontent.com/karpathy/ng-video-lecture/master/input.txt'
text = requests.get(url).text
sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[.?!)\n])\s', text)
sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
corpus = sentences[:10000]
with open('tiny_shakespeare_10k.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(corpus))
print(f'Loaded {len(corpus):,} sentences')

# 2. Training functions (same as previous example)...
# [Full training code here - abbreviated for this call]
print('Run this script to train all tokenizers!')