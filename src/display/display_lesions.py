import matplotlib.pyplot    as plt
import numpy                as np
import pickle
import os
from utils import clear_directory_content
def plot(stuff, label, subject_id, course, roi):
    r = len(stuff)  # Number of rows (types of images)
    indexes = np.where(np.sum(stuff[1], axis=(1, 2)) > 0)[0]  # Find non-empty slices more efficiently
    c = 6 if len(indexes) >= 6 else len(indexes)  # Number of columns (slices)
    indexes = np.linspace(min(indexes), max(indexes), c+2, dtype=int)[1:-1]
    labels = ['MRI', 'RTD']
    
    fig = plt.figure(figsize=(13, 8))  # Optional: Smaller figure size
    fig.suptitle(f'Subject: {subject_id} Course: {course}', fontsize=12, y=0.95)  
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)  # Adjust values as needed
    
    fig.suptitle(f'Subject: {subject_id} Course: {course} Roi: {roi} Label: {label}', fontsize=12)

    for i, s in enumerate(stuff):
        images = s[indexes]
        for j, img in enumerate(images):
            ax = fig.add_subplot(r, c, i * c + j + 1)  # Create subplot
            ax.axis('off')
            ax.imshow(img)
            if j == 0:
                ax.set_title(f'{labels[i]}', fontsize=10)
    if label == 'recurrence':
        plt.savefig(os.path.join('.', 'img_lesions', 'recurrence', f'{subject_id}_{course}_{roi}.png'))
    else:
        plt.savefig(os.path.join('.', 'img_lesions', 'stable', f'{subject_id}_{course}_{roi}.png'))
        
    plt.close()

def display(plt, axes, images, cur_row, c):
    for i in range(c):
        axes[cur_row, i].imshow(images[i])
        axes[cur_row, i].set_axis_off()


if __name__ == '__main__':
    path_to_data_folder = os.path.join('.', 'lesions_cropped') 
    clear_directory_content(os.path.join('.', 'img_lesions'))
    
    os.makedirs(os.path.join('.', 'img_lesions', 'stable'), exist_ok=True)
    os.makedirs(os.path.join('.', 'img_lesions', 'recurrence'), exist_ok=True)
    
    file_names = [f for f in os.listdir(path_to_data_folder)]
    
    for f in file_names:
        subject_id, course, roi = f.split('_')
        roi = roi.split('.')[0]
        with open(os.path.join(path_to_data_folder, f), 'rb') as input_file:
            data = pickle.load(input_file)
            plot([data['mr'], data['rtd']], data['label'], subject_id, course, roi)