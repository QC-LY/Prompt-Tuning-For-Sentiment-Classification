cd ../
python prompt_train.py --train_data_path ./data/mydata/new_train.json \
--train_label_path ./data/mydata/new_train_labels.json \
--dev_data_path ./data/mydata/dev.json \
--dev_label_path ./data/mydata/dev_labels.json \
--model_name_or_path t5 \
--output_dir ./checkpoint