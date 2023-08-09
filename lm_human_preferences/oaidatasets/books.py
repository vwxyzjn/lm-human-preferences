import random
from datasets import load_dataset


def books_generator(mode, seed=0, shuffle=False, comm=None):
    dataset = load_dataset("bookcorpus", split=mode)

    if shuffle:
        random.seed(seed)
        dataset = dataset.shuffle(seed)

    for i, data in enumerate(dataset):
        if i == 0:
            print("===========", data["text"])
        text = data["text"]
        yield text