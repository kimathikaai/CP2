# Setup
1. Create a virtualenv environment and install the required dependencies
```bash
cd ~/
virtualenv  -p /usr/bin/python3.8 myenv
# Activate the virtual environment
source myenv/bin/activate
# Clone the repository
git clone https://github.com/kimathikaai/CP2.git
# Install the required packages
pip install -r CP2/requirements.txt
```
2. Setup wandb for remote logging. Ensure you have access to the correct team and project
```bash
# After installing wandb via the `requirements.txtx` log in and paste your API key
wandb login
```
3. This repository includes the `mmsegmentation` submodule which will also need to be setup
```bash
# Install mmcv using mim
pip install -U openmim
mim install mmengine
mim install 'mmcv>=2.0.0'
# Install mmsegmentation from the source
cd ~/CP2/mmsegmentation
pip install -v -e .
# '-v' means verbose, or more output
# '-e' means installing a project in editable mode
# thus any local modifications made to the code will take effect without reinstallation
```
4. Join the `#server` slack channel to communicate what GPUs you'll be using and use nvitop to monitor GPU utilization
```bash
# Installed through the requirements.txt file
nvitop
```
5. Get familiar with using [tmux](https://hamvocke.com/blog/a-quick-and-easy-guide-to-tmux/) to run your processes on the server

# Pre-training and Fine-tuning
Pre-training and fine-tuning is done on multiple GPUs. Before running a script make sure the GPUs you're using are available and specified using the `CUDA_VISIBLE_DEVICES` environment variable.
```bash
# Pre-training and fine-tuning on the polpy datasets
./scripts/polyp.sh
```

# Citation
```
@article{wang2022cp2,
  title={CP2: Copy-Paste Contrastive Pretraining for Semantic Segmentation},
  author={Wang, Feng and Wang, Huiyu and Wei, Chen and Yuille, Alan and Shen, Wei},
  journal={arXiv preprint arXiv:2203.11709},
  year={2022}
}
```
