import numpy as np
import os
from itertools import groupby
import tqdm
import pickle

def write_graph_from_transcripts(dataset, bg_class=[0], map_delimiter=' '):
    """
    Generate and write a task graph from transcripts of actions.

    Parameters:
    - dataset (str): The name of the dataset.
    - bg_class (list): List of background class labels to ignore.
    - map_delimiter (str): Delimiter used in the mapping file.
    """
    gt_path = os.path.join('data', dataset, 'groundTruth')
    mapping_file = os.path.join("data", dataset, "mapping.txt")
    graph_path = os.path.join('data', dataset, 'graph')
    os.makedirs(graph_path, exist_ok=True)
    
    # Create a dictionary to map actions to indices
    actions_dict = dict()
    with open(mapping_file, 'r') as f:
        for line in f:
            actions = line.strip().split(map_delimiter)
            actions_dict[actions[1]] = int(actions[0])
    
    # Initialize matrices for predecessor and successor relationships
    pre_mat = np.zeros([len(actions_dict), len(actions_dict)])
    suc_mat = np.zeros([len(actions_dict), len(actions_dict)])
    count = np.zeros([len(actions_dict)])
    
    # Process each video in the ground truth path
    for vid in tqdm.tqdm(os.listdir(gt_path)):
        file_ptr = open(os.path.join(gt_path, vid), 'r')
        content = file_ptr.read().split('\n')[:-1]
        classes = np.zeros([len(content)], dtype=np.int32)
        
        # Map each action in the content to its corresponding index
        for i in range(len(classes)):
            classes[i] = actions_dict[content[i]]
        
        # Filter out background classes
        classes_wo_bg = [a for a in classes if a not in bg_class]
        transcript = [k for k, v in groupby(classes_wo_bg)]
        
        # Update counts and matrices
        for a in transcript:
            count[a] += 1
        for pre_action, suc_action in zip(transcript[:-1], transcript[1:]):
            pre_mat[pre_action, suc_action] += 1
            suc_mat[suc_action, pre_action] += 1
    
    # Normalize the matrices
    # after normalization, pre_mat and suc_mat are not symmetric
    pre_mat = pre_mat / np.maximum(count[None, :], 1e-5)
    suc_mat = suc_mat / np.maximum(count[None, :], 1e-5)
    
    # Save the graph
    graph = {'matrix_pre': pre_mat, 'matrix_suc': suc_mat}
    with open(os.path.join(graph_path, 'graph.pkl'), 'wb') as f:
        pickle.dump(graph, f)
    print(f"Finished writing graph for {dataset} in {graph_path}")

if __name__ == '__main__':
    # Generate graphs for specified datasets
    write_graph_from_transcripts('gtea', [10], ' ')
    for dataset in ['coffee', 'tea', 'pinwheels', 'oatmeal', 'quesadilla']:
        write_graph_from_transcripts(dataset, [0], '|')

