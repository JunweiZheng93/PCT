# Setup
Python version: 3.9 \
To install all dependencies, enter:
```bash
pip install -r requirements.txt
```
Be careful, please install PyTorch using this command:
```bash
conda install pytorch=1.11.0 cudatoolkit=11.3 -c pytorch
```

# How to run
When using `AnTao350M` dataset, run:
```bash
python train_shapenet.py datasets=shapenet_AnTao350M usr_config=YOUR/USR/CONFIG/PATH
```
When using `Yi650M` dataset, run:
```bash
python train_shapenet.py datasets=shapenet_Yi650M usr_config=YOUR/USR/CONFIG/PATH
```

# About configuration files
The train/test script will read the default configuration file (`./configs/default.yaml`) and the user specified 
configuration file before training/testing. It is not recommended to modify the default file directly. It is encouraged 
to create a new yaml file and specify the file via `usr_config` argument. For example, in your usr config file, enter:
```yaml
train:
  dataloader:
    batch_size: 32
  lr_scheduler:
    which: stepLR
    stepLR:
      gamma: 0.5
      decay_step: 50
```
Then run the train script:
```bash
python train_shapenet.py datasets=shapenet_AnTao350M usr_config=YOUR/USR/CONFIG/PATH
```
Check the default configuration file for all legal hyper-parameters.

# WandB (Weights and biases)
We use wandb to log all experiment results. It is an amazing logger for deep learning. If you want to disable the wandb 
logger, do it in your usr config:
```yaml
wandb:
  enable: false
```
Otherwise, you need to [create a wandb account](https://wandb.auth0.com/login?state=hKFo2SBjQURna1lEV1Jsb2dDUE5GYjZobGU4dUVmZEVRSzNGR6FupWxvZ2luo3RpZNkgU1pRX1JoM2RGQUw5WVRDZnNKeTJWYVV2b0xyX21HUDKjY2lk2SBWU001N1VDd1Q5d2JHU3hLdEVER1FISUtBQkhwcHpJdw&client=VSM57UCwT9wbGSxKtEDGQHIKABHppzIw&protocol=oauth2&nonce=cENYVVVPNC5MRWNmN3lGdw%3D%3D&redirect_uri=https%3A%2F%2Fapi.wandb.ai%2Foidc%2Fcallback&response_mode=form_post&response_type=id_token&scope=openid%20profile%20email&signup=true) first. 
Click the `Settings` and generate your API Key:
![user setting](./figures/usr_setting.png)
![api_key](./figures/api_key.png)
Then back to your usr config:
```yaml
wandb:
  enable: true
  api_key: your api key here  # your wandb api key
  entity: kit-ies  # the place to save your runs. can be your wandb username or team name
  project: pct  # the name of your project
  name: my_experiment  # the name of your run
```