# Most imortant scripts and folders

### python3 data_reader.py --debug

It reads the raw medical data present inside data/origin/ . The final result is saved in the data/processed/global_data.pkl which consists of a dicitonary containing clinical records, cropped MRI, cropped radiation dose maps, subject ids and labels. The debug flag, if set, selects only one subject: used for debugging.

### python3 grid_search_keep_test.py --config configs/base_model.yaml --experiment_name basemodelexp1 --k 5

Different parameters are tested by using the same test set of the article. Parameters are chosen from the params_grid.py file that overwrite the base configuration passed as an argument. By setting k to 5 a stratified 5 fold cross validation is used to produce 5 different train/validation splits

### python3 grid_search_whole_dataset.py --k 10 --config configs/base_model.yaml --experiment_name basemodelexp1

Different parameters are tested by cross validating the whole dataset. Parameters are chosen from the params_grid.py file that overwrite the base configuration passed as an argument. By setting k to 10 a stratified 10 fold cross validation is used to produce 10 different train/test splits. 

### python3 train_classifier.py --config configs/base_model.yaml --experiment_name basemodelexp1 --version 0

Used mainly for debugging, a static split is used. The chosen configuration is passed as argument

### python3 display_lesions.py

Save inside display/lesions/ the processed lesions saved in data/ptocessed/global_data.pkl

### modules/classification_module.py
Handles the classification process. Specifically, set the chosen model, handles the training, validation and test steps and epochs, set the optimizer...

### models/
Developed models

### configs/
Models base configurations

### datasets/
Dataloaders

### log/
Models checkpoints

Link to the brain mri dataset:\
https://www.cancerimagingarchive.net/collection/brain-tr-gammaknife/