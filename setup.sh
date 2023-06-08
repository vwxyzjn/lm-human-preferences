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
conda create -n myenv -c conda-forge tensorflow-gpu=1.15
conda activate myenv

poetry install
poetry run pip install tensorflow-gpu==1.13.1
poetry run pip install horovod==0.18.1

experiment=descriptiveness
reward_experiment_name=testdesc-$(date +%y%m%d%H%M)
poetry run python launch.py train_reward $experiment $reward_experiment_name


