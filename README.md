# Prompt Tuning For Sentiment Classification Base On Pre-trained Language Models

## **Code for the AI Internship at Northeastern University**

```
|--checkpoint/
|--data/
|------train_demo.json
|------train_labels_demo.json
|--commands/
|------train.sh
|------train_t5.sh
|------train_bert_bitfit.sh
|------train_bert_prompt.sh
|------train_t5_bitfit.sh
|------train_t5_prompt.sh
|--output/
|--bert_model.py
|--data_gain.py
|--data_ultis.py
|--inference.py
|--KB_prompt.py
|--prompt_bert.py
|--prompt_t5.py
|--prompt_test.py
|--prompt_train.py
|--t5_model.py
|--test.py
|--train.py
|--utils.py
|--README.md
```

## Data

The data for this project is internal. The format of the data is as shown in the example in the `./data/train.json` and `./data/train_labels.json` . 

The dev set and the test set have the same format as the train set.

Before you can run the code, you will need to process your own dataset into the following 6 files in `./data/`:

```
|--data/
|------train_demo.json
|------train_labels_demo.json
|------train.json
|------train_labels.json
|------dev.json
|------dev_labels.json
|------test.json
|------test_labels.json
```

## Experiment

We explore the different performance of pre-trained language model(BERT and T5) among Full Fine-tuning, Bias-term Fine-tuning and Prompt-tuning. We regard ***accuracy*** as our main evaluation.

| Method             | ACC       |
| ------------------ | --------- |
| BERT-fine-tuning   | 85.66     |
| BERT-BitFit        | 83.41     |
| BERT-hard-P-tuning | 84.11     |
| BERT-soft-P-tuning | 85.09     |
| T5-fine-tuning     | **87.03** |
| T5-BitFit          | 83.55     |
| T5-hard-P-tuning   | 85.55     |
| T5-soft-P-tuning   | **86.48** |

## Commands

```shell
cd ./commands
```

### 1. Full fine tuning

```shell
# For BERT
bash train.sh
# For T5
bash train_t5.sh
```

### 2. Bias-term Fine-tuning(BitFit)

```shell
# For BERT
bash train_bert_bitfit.sh
# For T5
bash train_t5_bitfit.sh
```

### 3. Prompt-tuning

```shell
# For BERT
bash train_bert_prompt.sh
# For T5
bash train_t5_prompt.sh
```

### 4. Inference for test

```shell
# Fine tuning or BitFit
python test.py --model_name bert/t5
# prompt tuning
python prompt_test.py --model_name bert/t5
```

## Others

checkpoints will be saved in ./checkpoints

train and test logs will be saved in ./checkpoints

If you have questions, suggestions, and bug reports, please email:

```
lvyuanhuiyi@foxmail.com
```

