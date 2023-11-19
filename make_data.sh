mkdir data/combined_data
cp data/data_01/* data/combined_data/
cp data/data_02/* data/combined_data/
python convert_yolox.py data/combined_data data/annotations.json
python cocosplit.py --having-annotations --multi-class -s 0.9 data/annotations.json data/train.json data/val.json