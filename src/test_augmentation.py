import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from utils.augment import *
from utils.utils import clear_directory_content

def test_augmentation():
    path_to_data_folder = os.path.join(os.path.dirname(__file__),'..', 'data', 'processed')
    file_names = [f for f in os.listdir(path_to_data_folder) if 'train' in f]
    #augmentations = [shear, brightness, flip, elastic]
    augmentations = [random_affine]
    #augmentation_names = ['shear', 'brightness', 'flip', 'elastic']
    augmentation_names = ['random_affine']
    OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'display', 'augmentation')
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    clear_directory_content(OUTPUT_PATH)
    
    for file_split in file_names:
        if file_split.endswith('.pkl'):
            with open(os.path.join(path_to_data_folder, file_split), 'rb') as input_file:
                data = pickle.load(input_file)
                chosen_samples = np.argsort([np.sum(m.cpu().detach().numpy() > 0) for m in data['mr']])[-6:-1]
                MAX_SAMPLES = len(chosen_samples)
                for cnt, i_sample in enumerate(chosen_samples):
                    print(f"{cnt+1}/{MAX_SAMPLES}", end='\r')
                    mr, rtd = data['mr'][i_sample], data['rtd'][i_sample]
                    
                    for i_aug, augment in enumerate(augmentations):
                        mr_aug, rtd_aug = augment(mr, rtd)
                        stuff = [mr.cpu().detach().numpy(), mr_aug.cpu().detach().numpy(), rtd.cpu().detach().numpy(), rtd_aug.cpu().detach().numpy()]
                        
                        c = 6 # if len(indexes) >= 6 else len(indexes)  # Number of columns (slices)
                        
                        fig, axes = plt.subplots(4, c, figsize=(20, 10))
                        
                        fig.suptitle(augmentation_names[i_aug])
                        
                        for i, s in enumerate(stuff):
                            axes[i, 0].set_title(['MRI', 'MRI augmented', 'RTD', 'RTD augmented'][i])
                            
                            indexes = np.where(np.sum(stuff[i], axis=(1, 2)) > 0)[0]
                            indexes = np.linspace(min(indexes), max(indexes), c+2, dtype=int)[1:-1]
                            
                            for j in range(len(indexes)):
                                axes[i, j].imshow(s[:, :, indexes[j]])
                                axes[i, j].axis('off')
                    
                        plt.savefig(os.path.join(os.path.dirname(__file__), 'display', 'augmentation', f'{i_sample}_{augmentation_names[i_aug]}.png'), dpi=300)
                        plt.close()
                        
    print("Done!", end='\r')

if __name__ == '__main__':
    test_augmentation()

