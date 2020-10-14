


![GeDi logo](https://github.com/salesforce/GeDi/blob/master/gedi_logo_small.png)
--------------------------------------------------------------------------------

Official implementation of [GeDi: Generative Discriminator Guided Sequence Generation](https://arxiv.org/abs/2009.06367)

Blogpost [here](https://blog.einstein.ai/gedi/)


#### Colab Notebook on controlling topic using GeDi [here](https://colab.research.google.com/github/salesforce/GeDi/blob/master/GeDi_guided_GPT_2_XL.ipynb)     

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salesforce/GeDi/blob/master/GeDi_guided_GPT_2_XL.ipynb)

## Updates

Sept 29, 2020: Adding support for [GeDi-guided GPT-3 generation](https://github.com/salesforce/gedi#gpt-3-generation-added-after-paper-api-access-needed) (API key needed)


## Introduction

GeDi is a method of using class-conditional language models (which we refer to as generative discriminators (GeDis)) to guide generation from other (potentially much larger) language models. This has several advantages over finetuning large language models directly including:

* significantly less training computation.
* maintaining the diversity of the original language model (If we finetune a large pretrained language model to a specific attribute dataset, we will likely reduce the broad generation capabilities of the model).
* teaching the language model what not to generate. This is especially useful for applications like detoxification.


GeDi is a form of discriminator guided generation. A discriminator that can classify an attribute could be used to guide language model generation towards that attribute by classifying the sequences that result from candidate next tokens. However, using a normal discriminator (such as [BERT](https://arxiv.org/abs/1810.04805)) to do this would be very computationally expensive during generation, since it would require feeding in every candidate next token one-by-one to the discriminator to be classified. However, using generative discriminators, we can very efficiently classify candidate next tokens during generation using Bayes rule (see Section 3.1 of the paper). As an added bonus, [generative discriminators can be used as zero shot classifiers](https://arxiv.org/abs/1703.01898), and can therefore be used to guide generation towards unseen topics.


## Dependencies

- Python 3.7, PyTorch 1.4
(We recommend creating a container using the [pytorch/pytorch:1.4-cuda10.1-cudnn7-devel](https://hub.docker.com/layers/pytorch/pytorch/1.4-cuda10.1-cudnn7-devel/images/sha256-c612782acc39256aac0637d58d297644066c62f6f84f0b88cfdc335bb25d0d22?context=explore) official pytorch docker image.)
- Run `scripts/setup.sh`:
  ```
  cd scripts
  bash setup.sh
  ```
  This will install the following:

  - Transformers v2.8.0 and its example-specific requirements mentioned [here](https://github.com/huggingface/transformers/tree/master/examples#important-note)
  - (optional) Apex (details [here](https://github.com/NVIDIA/apex#linux)) for fp16 training and generation (installing apex takes a while; comment corresponding lines in `setup.sh` if you want to skip)
  - `wget` and `unzip` (to download and unzip data and model checkpoints)

## Generating from models used in paper
- First download the models:
```
cd scripts
bash get_models.sh
```
This downloads and saves the topic, sentiment, and detoxifier models in the folder `../pretrained_models`

- To generate, use `bash run_generation.sh`, which calls `../generate_GeDi.py` with the appropriate arguments (set for topic generation by default).

Important arguments include:

* `--mode` can be set to `topic`, `sentiment`, or `detoxify`
* `--gen_type` can be set to `gedi` for GeDi guided generation, `cclm` for class conditional generation, or `gpt2` to generate from raw GPT-2
* `--gen_length` max length of generation
* `--gedi_model_name_or_path` path to GeDi model. If unused, will assume you ran `bash get_models.sh` and infer model directory from `--mode` argument
* `--filter_p` equal to 1 - \rho in Equation 7 of the paper
* `--target_p` equal to \tau from the paper
* `--disc_weight` exponent for posterior weighting (\omega in Equation 6 of the paper)
* `--fp16` converts GPT2-XL weights to fp16 for faster generation and less GPU memory usage

Running will allow you to enter control codes and prompts for generation in a continuous loop until you exit.

### Topic generation (Section 6.3 & 6.4 of the paper)
- Set `--mode topic` in `scripts/run_generation.sh`
- You will be prompted to give a topic code. The model was trained on `world`, `sports`, `business`, and `science`, but can often generate other topics zero-shot, for instance `space`, `fire`, `climate`, `education`
- If the topic code you give is more than one [BPE token](https://arxiv.org/abs/1508.07909), the model often struggles because the 4 training topics were all 1 BPE token. You will be warned that this might not work, but can proceed by hitting enter again (or can type a new topic code).
- After the topic code, you will be asked to give a prompt to the model to condition on for generation.

### Sentiment control (Section 6.1 of the paper)
- Set `--mode sentiment` in `scripts/run_generation.sh`
- The model can controllably generate positive or negative text. When generalizing to other domains such as stories, this often translates to positive/negative mood or tone of the story (since sentiment implies an opinion).
- The model is set to positive sentiment by default. You will be prompted for the opportunity to change to negative sentiment by typing `n`. Note that the negative model can be very negative, and this sometimes results in toxic or offensive samples.
- You will then be asked to give a prompt to the model to condition on for generation.

### Detoxication (Section 6.2 of the paper)
- Set `--mode detoxify` in `scripts/run_generation.sh`
- This mode can be used to avoid generating toxic or offensive text.
- You will then be asked to give a prompt to the model to condition on for generation.
- GeDi can often find a way to navigate especially aggressive prompts, but does rarely but occasionally still generate toxic text if given certain prompts. We observed this can be a problem for longer generations.

### Class-conditional LM and GPT-2 generation
- Two of the baselines we consider are generating from GPT-2 (will give same result regardless of control codes), and generating from the GeDi model directly as a class-conditional language model (instead of using it to guide generation from GPT-2).
- Set `--gen_type gpt2` to generate from GPT-2, and `--gen_type cclm` to generate directly from the GeDi as a class-conditional language model. `--gen_type cclm` corresponds to all experiments in Section 5 of the paper, and the CC-LM baselines in Section 6.1.

### GPT-3 generation (added after paper, API access needed)
- If you have your own GPT-3 API secret key, you can use GeDi to guide decoding from GPT-3.
- This is somewhat limited, since the GPT-3 API only allow access to the top 100 next token log probabilities.
- Reuses settings for controlling GPT-2 (which uses all next token log probs), retuning for GPT-3 could give better results.
- It is also slow (up to 1 second per token) because modifying GPT-3 decoding requires calling the API one token at a time.

To control sentiment from GPT-3 using your API key (should have prefix "sk"):

`pip install openai`

`python ../generate_GeDi.py --penalize_cond --gen_length 100 --mode sentiment --gpt3_api_key sk-xxxxxxxx`

You can also try changing the `--mode` or other arguments. To generate directly from GPT-3 without GeDi using our same greedy decoding scheme:

`python ../generate_GeDi.py --penalize_cond --gen_length 100 --mode sentiment --gen_type gpt2 --gpt3_api_key sk-xxxxxxx`


## Train your own GeDi

- This repository includes code to train a topic GeDi using GeDi training.
- There are some differences in this training script and the one used to train the pretrained model. The pretrained model only used half of AG news, and there were some slight differences in preprocessing.
- This runs in about 5 hours on a 16GB V100 GPU on GCP.
- First, download and process the topic data:

```
cd scripts
bash get_data.sh
```

- Then run training using:

`bash run_training.sh` which calls `../train_GeDi.py` with the appropriate arguments

- The directory for model to be saved is specified by `output_dir` argument.
- When generating from your trained GeDi, you will need to call `../generate_GeDi.py` (called from `bash run_generation.sh`) with `--gedi_model_name_or_path` set to the directory of your trained model.

## Citation
```
@article{KrauseGeDi2020,
  title={{GeDi: Generative Discriminator Guided Sequence Generation}},
  author={Krause, Ben and Gotmare, Akhilesh Deepak and McCann, Bryan and Keskar, Nitish Shirish and Joty, Shafiq and Socher, Richard and Rajani, Nazneen Fatema},
  journal={arXiv preprint arXiv:2009.06367},
  year={2020}
}
```

## License
The code is released under the BSD-3 License (see `LICENSE.txt` for details), but we also ask that users respect the following:

This software should not be used to promote or profit from violence, hate, and division, environmental destruction, abuse of human rights, or the destruction of people's physical and mental health.
