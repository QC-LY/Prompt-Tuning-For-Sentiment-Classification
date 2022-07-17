# Code for the AI Internship Course at Northeastern University

## environment

    pytorch == 1.8.0
    transformers >= 4.20.1

## commands

    cd ./commands

### 1. Full fine-tuning
full fine-tuning BERT:

    bash train.sh

full fine-tuning T5:

    bash train_t5.sh

### 2. Bitfit

Bitfit BERT or T5:

    bash train_bert_bitfit.sh  #BERT
    bash train_t5_bitfit.sh    #T5

### 3. Prompt-tuning

prompt-tuning BERT:

    bash train_bert_prompt.sh

prompt-tuning T5:

    bash train_t5_prompt.sh

test in fine-tune or bitfit model:

    python test.py --model_name bert/t5

test in prompt-tune model:

    python prompt_test.py --model_name bert/t5

## other
datasets are in 
    ./data

checkpoints will be saved in
    ./checkpoints

train and test logs will be saved in 
    ./checkpoints
