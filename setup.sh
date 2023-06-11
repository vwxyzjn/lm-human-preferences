# set up an AWS p3.16xlarge machine with ami-0869f17a1c79dad4c
pip3 install poetry==1.3.1
echo 'export PATH=~/.local/bin:$PATH' >> ~/.bashrc


# install pyenv
curl https://pyenv.run | bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
source ~/.bashrc

# setup conda
pyenv install miniconda3-4.7.12
pyenv activate miniconda3-4.7.12
conda create -y -n myenv -c anaconda tensorflow-gpu=1.14
conda activate myenv
conda create -y -n myenv2 -c anaconda tensorflow-gpu=1.13
conda activate myenv2


pip install "cloudpickle==1.2.1" "dataclasses==0.6.0" "fire==0.1.3" "ftfy==5.4.1" "mpi4py==3.0.2" "mypy==0.580" "numpy==1.16.2" "pytest-instafail==0.3.0" "pytest-timeout==1.2.0" "pytest==3.5.0" "pytz==2019.1" "regex==2017.4.5" "requests==2.18.0" "tqdm==4.31.1" "typeguard>=2.2.2"
pip install "google-api-python-client==1.7.8" "google-cloud-storage==1.13.0"
pip install datasets

experiment=descriptiveness
reward_experiment_name=testdesc-$(date +%y%m%d%H%M)


mkdir -p gpt-2/encodings
wget -O gpt-2/encodings/encoder.json https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
wget -O gpt-2/encodings/vocab.bpe https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
mkdir -p gpt-2/models/124M/
wget -O gpt-2/models/124M/checkpoint https://openaipublic.blob.core.windows.net/gpt-2/models/124M/checkpoint
wget -O gpt-2/models/124M/encoder.json https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json
wget -O gpt-2/models/124M/hparams.json https://openaipublic.blob.core.windows.net/gpt-2/models/124M/hparams.json
wget -O gpt-2/models/124M/model.ckpt.data-00000-of-00001 https://openaipublic.blob.core.windows.net/gpt-2/models/124M/model.ckpt.data-00000-of-00001
wget -O gpt-2/models/124M/model.ckpt.index https://openaipublic.blob.core.windows.net/gpt-2/models/124M/model.ckpt.index
wget -O gpt-2/models/124M/model.ckpt.meta https://openaipublic.blob.core.windows.net/gpt-2/models/124M/model.ckpt.meta
wget -O gpt-2/models/124M/vocab.bpe https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe

# USE_TORCH=1 because `datasets` will try to do something with tensorflow dataset which causes a bad interaction
USE_TORCH=1 python launch.py train_reward $experiment $reward_experiment_name
USE_TORCH=1 mpiexec -n 1 python -c 'import sys; import pickle; pickle.loads(open("/tmp/pickle_fn", "rb").read())()'

bookcorpus
https://huggingface.co/datasets/bookcorpus

Daily mail, len(287227)
https://huggingface.co/datasets/cnn_dailymail


TLDR: len(datas) = 3053579
https://huggingface.co/datasets/webis/tldr-17

# ensure GPU works
python -c 'from tensorflow.python.client import device_lib; print(device_lib.list_local_devices())'


wget https://openaipublic.blob.core.windows.net/lm-human-preferences/tldr/train-subset.json

git config user.name "Costa Huang"
git config user.email "costa.huang@outlook.com"
