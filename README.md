# Transfer Knowledge from Natural Language to Electrocardiography: Can We Detect Cardiovascular Disease Through Language Models? 

##### Accepted by EACL 2023

## Citation

If you feel our code or models helps in your research, kindly cite our papers:

```
@article{Qiu2023TransferKF,
  title={Transfer Knowledge from Natural Language to Electrocardiography: Can We Detect Cardiovascular Disease Through Language Models?},
  author={Jielin Qiu and William Han and Jiacheng Zhu and Mengdi Xu and Michael Rosenberg and Emerson Liu and Douglas Weber and Ding Zhao},
  journal={ArXiv},
  year={2023},
  volume={abs/2301.09017}
}
```


## Dataset

Use the PTB-XL dataset: https://physionet.org/content/ptb-xl/1.0.2/

## Preprocessing

Use the script ecg_preprocess.ipynb

## Extract ECG features

Use the script ecg_extract_features.ipynb

(Note the current data is based on the PTB-XL database).


## Set up Environment

Create a virtual environment and activate it. 

```
python -m venv .env
source .env/bin/activate
```

Install basic requirements.

```
pip install -r requirements.txt
```

## Translation

The original ECG reports from the PTB-XL dataset is in German. We use EasyNMT('opus-mt') model for translation. 
Please translate the data before training with the `translate.py` file.
Please view the `config.py` file in tandem and customize as necessary. 

## Training 

The `main.py` file is used for training selected models.
Please view the `config.py` file in tandem and customize as necessary. 

## Finetuneing

The `finetune.py` file is used for finetuning the LLM model. This step is necessary to reproduce the paper results. We finetuned the LLM for 50 epochs.
Please view the `config.py` file in tandem and customize as necessary. 

## Evaluation

During training in the `main.py` file, we do include an evaluation step but if you wanted to only run inference on the models,
please use `evaluate.py`. 
Please view the `config.py` file in tandem and customize as necessary. 

### Contact

If you have any question, please contact wjhan@andrew.cmu.edu, jielinq@andrew.cmu.edu.
