import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from utils.augment import *
from utils.utils import clear_directory_content

def test_augmentation():
    MAX_SAMPLES = 5
    path_to_data_folder = os.path.join(os.path.dirname(__file__),'..', 'data', 'processed')
    file_names = [f for f in os.listdir(path_to_data_folder) if 'train' in f]
    #augmentations = [shear, brightness, flip, elastic]
    augmentations = [gaussian_noise]
    #augmentation_names = ['shear', 'brightness', 'flip', 'elastic']
    augmentation_names = ['gaussian_noise']
    OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'display', 'augmentation')
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    chosen_samples = [137, 68, 160, 192, 28]
    clear_directory_content(OUTPUT_PATH)
    
    for file_split in file_names:
        if file_split.endswith('.pkl'):
            with open(os.path.join(path_to_data_folder, file_split), 'rb') as input_file:
                data = pickle.load(input_file)
                #numbers = list(range(len(data['mr'])))
                for cnt, i_sample in enumerate(chosen_samples):
                    print(f"{cnt+1}/{MAX_SAMPLES}", end='\r')
                    mr, rtd = data['mr'][i_sample], data['rtd'][i_sample]
                    
                    indexes = np.where(np.sum(mr.cpu().detach().numpy(), axis=(1, 2)) > 0)[0]  # Find non-empty slices more efficiently
                    c = 6 if len(indexes) >= 6 else len(indexes)  # Number of columns (slices)
                    indexes = np.linspace(min(indexes), max(indexes), c+2, dtype=int)[1:-1]
                    
                    for i_aug, augment in enumerate(augmentations):
                        mr_aug, rtd_aug = augment(mr, rtd)
                        stuff = [mr.cpu().detach().numpy(), mr_aug.cpu().detach().numpy(), rtd.cpu().detach().numpy(), rtd_aug.cpu().detach().numpy()]
                        
                        fig, axes = plt.subplots(4, c, figsize=(20, 10))
                        
                        fig.suptitle(augmentation_names[i_aug])
                        
                        for i, s in enumerate(stuff):
                            axes[i, 0].set_title(['MRI', 'MRI augmented', 'RTD', 'RTD augmented'][i])
                            for j in range(c):
                                axes[i, j].imshow(s[:, :, indexes[j]])
                                axes[i, j].axis('off')
                    
                        plt.savefig(os.path.join(os.path.dirname(__file__), 'display', 'augmentation', f'{i_sample}_{augmentation_names[i_aug]}.png'), dpi=300)
                        plt.close()
                        
    print("Done!", end='\r')

if __name__ == '__main__':
    test_augmentation()

