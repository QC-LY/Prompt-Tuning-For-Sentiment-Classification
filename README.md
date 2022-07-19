# Code for the AI Internship Course at Northeastern University

    |--checkpoint/
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


## environment

    # pytorch and transformers
    pytorch == 1.8.0
    transformers >= 4.20.1
    # OpenPrompt using Pip:
    pip install openprompt
    # OpenPrompt using Git:
    git clone https://github.com/thunlp/OpenPrompt.git
    cd OpenPrompt
    pip install -r requirements.txt
    python setup.py install


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
