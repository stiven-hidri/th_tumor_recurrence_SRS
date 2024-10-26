import argparse
import matplotlib.pyplot    as plt
import numpy                as np
import pickle
import os
from utils import clear_directory_content

def plot(stuff, label, progressive):
    r = len(stuff)  # Number of rows (types of images)
    indexes = np.where(np.sum(stuff[1], axis=(1, 2)) > 0)[0]  # Find non-empty slices more efficiently
    c = 6 if len(indexes) >= 6 else len(indexes)  # Number of columns (slices)
    indexes = np.linspace(min(indexes), max(indexes), c+2, dtype=int)[1:-1]
    labels = ['MRI', 'RTD','WDT_FUSION']
    
    fig = plt.figure(figsize=(13, 8))  # Optional: Smaller figure size
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)  # Adjust values as needed
    
    fig.suptitle(f'Label: {label}', fontsize=12)

    for i, s in enumerate(stuff):
        images = s[indexes]
        for j, img in enumerate(images):
            ax = fig.add_subplot(r, c, i * c + j + 1)  # Create subplot
            ax.axis('off')
            ax.imshow(img)
            if j == 0:
                ax.set_title(f'{labels[i]}', fontsize=10)
    if label == 1:
        plt.savefig(os.path.join(os.path.dirname(__file__), 'img_lesions', 'recurrence', f'figure_{progressive}.png'))
    else:
        plt.savefig(os.path.join(os.path.dirname(__file__), 'img_lesions', 'stable', f'figure_{progressive}.png'))
        
    plt.close()

def display(plt, axes, images, cur_row, c):
    for i in range(c):
        axes[cur_row, i].imshow(images[i])
        axes[cur_row, i].set_axis_off()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', action='store_true')
    args = parser.parse_args()
    if args.raw:
        path_to_data_folder = os.path.join(os.path.dirname(__file__), '..','..','data', 'raw') 
    else:
        path_to_data_folder = os.path.join(os.path.dirname(__file__), '..','..','data', 'processed') 

    clear_directory_content(os.path.join(os.path.dirname(__file__), 'img_lesions'))
    
    os.makedirs(os.path.join(os.path.dirname(__file__), 'img_lesions'), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), 'img_lesions', 'stable'), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), 'img_lesions', 'recurrence'), exist_ok=True)
    
    file_names = [f for f in os.listdir(path_to_data_folder) if 'train' in f]
    images, labels =  [], []
    
    for file_split in file_names:
        if file_split.endswith('.pkl'):    
            with open(os.path.join(path_to_data_folder, file_split), 'rb') as input_file:
                data = pickle.load(input_file)    
            images.extend([ [np.array(data['mr'][i]), np.array(data['rtd'][i]), np.array(data['mr_rtd_fusion'][i])] for i in range(len(data['mr']))])
            labels.extend(data['label'].squeeze().tolist())
    
    for i in range(10):
        print(f'\r{i+1}/{len(images)}', end='')
        plot(images[i], int(labels[i]), i)
