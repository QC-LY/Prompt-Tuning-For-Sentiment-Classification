cd ../
python train.py --train_data_path ./data/train.json \
--train_label_path ./data/train_labels.json \
--dev_data_path ./data/dev.json \
--dev_label_path ./data/dev_labels.json \
--model_name_or_path t5 \
--output_dir ./checkpoint \
--bitfit
