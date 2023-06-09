import hashlib
import os
import random
import re

import ftfy
from datasets import load_dataset


def clean_up_start(text):
    # if text[:2] == 'By':
    #     text = '\n'.join(text.split('\n')[2:])
    text = re.split(r'\(CNN\) +--', text)[-1]
    text = re.split(r"\(CNN\)", text[:100])[-1]+text[100:]
    text = re.sub(r"^and \w+\n", "", text)
    text = re.split(r".*UPDATED:\s+[0-9]{2}:[0-9]{2}.*[2011|2012|2013|2014|2015]", text)[-1]
    text = text.replace('’', "'")
    text = text.replace('‘', "'")
    return text.strip()

def cnndm_generator(mode, seed=0, shuffle=False, comm=None):
    dataset = load_dataset("cnn_dailymail", version="3.0.0", split=mode)

    if shuffle:
        random.seed(seed)
        dataset = dataset.shuffle(seed)

    for _, data in enumerate(dataset):
        original_text = data["article"]
        text = clean_up_start(original_text)
        text = ftfy.fix_text(text)

        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.split('@highlight')[0].strip()

        yield text
