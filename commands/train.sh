cd ../
python train.py --train_data_path ./data/new_train.json \
--train_label_path ./data/new_train_labels.json \
--dev_data_path ./data/dev.json \
--dev_label_path ./data/dev_labels.json \
--model_name_or_path bert \
--output_dir ./checkpoint
