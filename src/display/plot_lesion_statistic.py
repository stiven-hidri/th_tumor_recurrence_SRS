import matplotlib.pyplot    as plt
import numpy                as np
import pickle
import os
from utils import clear_directory_content

def display(plt, axes, images, cur_row, c):
    for i in range(c):
        axes[cur_row, i].imshow(images[i])
        axes[cur_row, i].set_axis_off()

if __name__ == '__main__':
    path_to_data_folder = os.path.join('.', 'lesions_cropped') 
    #clear_directory_content(os.path.join('.', 'img_lesions_statistics'))
    
    file_names = [f for f in os.listdir(path_to_data_folder)]
    
    volumes = []
    labels = []
    
    for f in file_names:
        subject_id, course, roi = f.split('_')
        with open(os.path.join(path_to_data_folder, f), 'rb') as input_file:
            data = pickle.load(input_file)
            vol = data['mr'].shape[0]*data['mr'].shape[2]
            label = data['label']
            
        if vol <100000:
            volumes.append(vol)
            labels.append(1 if label == 'recurrence' else 0)
        
    volumes = np.array(volumes)
    labels = np.array(labels)    
    
    plt.figure(figsize=(12,8))  # Adjust the size as needed

    for label in [1,0]:
        indexes = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(volumes[indexes], np.zeros(len(indexes)) + label, c= 'r' if label == 1 else 'g', marker='o', alpha=0.5)

    # plt.scatter(volumes, labels, cmap='RdYlGn', marker='o')  

    plt.xlabel('wide')
    plt.title('wide - recurrence')
    plt.legend(['recurrence', 'stable'])
    plt.savefig(os.path.join('.', 'img_lesions_statistics', 'wide-recurrence.png'))
    plt.show()
    plt.close()
    