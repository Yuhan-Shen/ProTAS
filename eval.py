#!/usr/bin/python2.7
# adapted from: https://github.com/colincsl/TemporalConvolutionalNetworks/blob/master/code/metrics.py

import numpy as np
import argparse
from itertools import groupby
import os
import json


def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
        f.close()
    return content


def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i + 1)
    return labels, starts, ends


def levenstein(p, y, norm=False):
    m_row = len(p)    
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], np.float)
    for i in range(m_row+1):
        D[i, 0] = i
    for i in range(n_col+1):
        D[0, i] = i

    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1] == p[i-1]:
                D[i, j] = D[i-1, j-1]
            else:
                D[i, j] = min(D[i-1, j] + 1,
                              D[i, j-1] + 1,
                              D[i-1, j-1] + 1)
    
    if norm:
        score = (1 - D[-1, -1]/max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score


def edit_score(recognized, ground_truth, norm=True, bg_class=["background"]):
    # modified edit_score to remove consecutive duplicates after filtering out background
    recognized_no_bg = [a for a in recognized if not a in bg_class]
    ground_truth_no_bg = [a for a in ground_truth if not a in bg_class]
    P = [k for k, g in groupby(recognized_no_bg)]
    Y = [k for k, g in groupby(ground_truth_no_bg)]
    #P, _, _ = get_labels_start_end_time(recognized, bg_class)
    #Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)


def f_score(recognized, ground_truth, overlap, bg_class=["background"]):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = 0
    fp = 0

    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)


def evaluate(dataset, result_dir, split, exp_id, num_epochs):
    ground_truth_path = "./data/"+dataset+"/groundTruth/"
    recog_path = result_dir #"./results/"+exp_id+"/"+dataset+"/epoch"+str(num_epochs)+"/split_"+split+"/"
    file_list = "./data/"+dataset+"/splits/test.split"+split+".bundle"

    list_of_videos = read_file(file_list).split('\n')[:-1]

    overlap = [.1, .25, .5]
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

    correct = 0
    total = 0
    correct_wo_bg = 0
    total_wo_bg = 0
    edit = 0
    map_delimiter = '|' if dataset in ['ptg', 'coffee', 'tea', 'pinwheels', 'oatmeal', 'quesadilla'] else ' '
    bg_class = ['BG'] if dataset in ['ptg', 'coffee', 'tea', 'pinwheels', 'oatmeal', 'quesadilla'] else ['background']

    for vid in list_of_videos:
        if not vid.endswith('.txt'):
            vid = vid + '.txt'
        gt_file = ground_truth_path + vid
        gt_content = read_file(gt_file).split('\n')[0:-1]
        
        recog_file = os.path.join(recog_path, vid.split('.')[0])
        recog_content = read_file(recog_file).split('\n')[1].split(map_delimiter)

        for i in range(len(gt_content)):
            if gt_content[i] not in bg_class:
                total_wo_bg += 1
                if gt_content[i] == recog_content[i]:
                    correct_wo_bg += 1
            total += 1
            if gt_content[i] == recog_content[i]:
                correct += 1
        
        edit += edit_score(recog_content, gt_content, bg_class=bg_class)

        for s in range(len(overlap)):
            tp1, fp1, fn1 = f_score(recog_content, gt_content, overlap[s], bg_class)
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1
            
    acc = 100*float(correct)/total
    acc_wo_bg = 100*float(correct_wo_bg)/total_wo_bg
    edit = (1.0*edit)/len(list_of_videos)
    res_list = [acc, acc_wo_bg, edit]

    #print("Acc: %.4f" % (100*float(correct)/total))
    #print('Edit: %.4f' % ((1.0*edit)/len(list_of_videos)))
    for s in range(len(overlap)):
        precision = tp[s] / float(tp[s]+fp[s])
        recall = tp[s] / float(tp[s]+fn[s])
    
        f1 = 2.0 * (precision*recall) / (precision+recall)

        f1 = np.nan_to_num(f1)*100
        #print('F1@%0.2f: %.4f' % (overlap[s], f1))
        res_list.append(f1)
    print(exp_id, ' '.join(['{:.2f}'.format(r) for r in res_list]))
    result_metrics = {'Acc': acc,  'Acc-bg': acc_wo_bg, 'Edit': edit, 
                    'F1@10': res_list[-3], 'F1@25': res_list[-2], 'F1@50': res_list[-1]}
    result_path = os.path.join(recog_path, 'split'+split+'.eval.json')
    with open(result_path, 'w') as fw:
        json.dump(result_metrics, fw, indent=4)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default="ptg")
    parser.add_argument('--split', default='1')
    parser.add_argument('--exp_id', default='default', type=str)
    parser.add_argument('--result_dir', default='', type=str)
    parser.add_argument('--num_epochs', default=100, type=int)

    args = parser.parse_args()
    evaluate(args.dataset, args.result_dir, args.split, args.exp_id, args.num_epochs)



if __name__ == '__main__':
    main()
