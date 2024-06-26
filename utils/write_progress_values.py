import numpy as np
import os
from itertools import groupby
import tqdm

def write_progress_values(dataset, bg_class=[0], map_delimiter=' '):
    """
    Generate and write progress values for each action in the dataset.

    Parameters:
    - dataset (str): The name of the dataset.
    - bg_class (list): List of background class labels to ignore.
    - map_delimiter (str): Delimiter used in the mapping file.
    """
    gt_path = os.path.join('data', dataset, 'groundTruth')
    mapping_file = os.path.join("data", dataset, "mapping.txt")
    progress_path = os.path.join('data', dataset, 'progress')
    os.makedirs(progress_path, exist_ok=True)
    
    # Create a dictionary to map actions to indices
    actions_dict = dict()
    with open(mapping_file, 'r') as f:
        for line in f:
            actions = line.strip().split(map_delimiter)
            actions_dict[actions[1]] = int(actions[0])
    
    # Process each video in the ground truth path
    for vid in tqdm.tqdm(os.listdir(gt_path)):
        file_ptr = open(os.path.join(gt_path, vid), 'r')
        content = file_ptr.read().split('\n')[:-1]
        classes = np.zeros([len(content)], dtype=np.int32)
        
        # Map each action in the content to its corresponding index
        for i in range(len(classes)):
            classes[i] = actions_dict[content[i]]
        
        # Initialize progress values array
        progress_values = np.zeros([len(actions_dict), len(content)])
        cur_frame = 0
        
        # Calculate progress values for each action segment
        for k, v in groupby(classes):
            segment_length = len(list(v))
            if k not in bg_class:
                cur_progress = (np.arange(segment_length) + 1) / segment_length
                progress_values[k, cur_frame:cur_frame+segment_length] = cur_progress
            cur_frame += segment_length
        
        # Save progress values to a file
        np.save(os.path.join(progress_path, vid[:-4]+'.npy'), progress_values)
    
    print(f"Finished writing progress values for {dataset} in {progress_path}")

if __name__ == '__main__':
    # Generate progress values for specified datasets
    write_progress_values('gtea', [10], ' ')
    for dataset in ['coffee', 'tea', 'pinwheels', 'oatmeal', 'quesadilla']:
        write_progress_values(dataset, [0], '|')

