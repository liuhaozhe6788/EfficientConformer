#!/bin/sh
python main.py --config_file configs/EfficientConformerCTCLarge.json --mode test-clean --initial_epoch swa-equal-401-450 
python main.py --config_file configs/EfficientConformerCTCLarge.json --mode test-other --initial_epoch swa-equal-401-450 
python main.py --config_file configs/EfficientConformerCTCMedium.json --mode test-clean --initial_epoch swa-equal-401-450 
python main.py --config_file configs/EfficientConformerCTCMedium.json --mode test-other --initial_epoch swa-equal-401-450 
python main.py --config_file configs/EfficientConformerCTCSmall.json --mode test-clean --initial_epoch swa-equal-401-450 
python main.py --config_file configs/EfficientConformerCTCSmall.json --mode test-other --initial_epoch swa-equal-401-450 