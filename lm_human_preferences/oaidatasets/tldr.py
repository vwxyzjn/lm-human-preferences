import random
from datasets import load_dataset


def tldr_generator(mode, seed=0, shuffle=False, comm=None):
    dataset = load_dataset("webis/tldr-17", split=mode)

    if shuffle:
        random.seed(seed)
        dataset = dataset.shuffle(seed)

    for _, data in enumerate(dataset):
        text = data["body"]
        yield text