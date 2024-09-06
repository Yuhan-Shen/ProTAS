# ProTAS

This is the repository for the paper [Progress-Aware Online Action Segmentation for Egocentric Procedural Task Videos](https://openaccess.thecvf.com/content/CVPR2024/papers/Shen_Progress-Aware_Online_Action_Segmentation_for_Egocentric_Procedural_Task_Videos_CVPR_2024_paper.pdf) accepted by CVPR 2024.

Most of the codes are adapted from [MSTCN](https://github.com/yabufarha/ms-tcn).

# Repository Structure

- **main.py**: Script to train and evaluate the model.
- **model.py**: Contains the implementation of the neural network models (MultiStageModel, SingleStageModel, etc.).
- **batch_gen.py**: Script for generating batches of data for training and evaluation.
- **eval.py**: Evaluation script.
- **utils**: Utility functions including `write_graph_from_transcripts` and `write_progress_values`.
- **data/**: Directory containing datasets, including ground truth and feature files.

# Data

**GTEA**: download GTEA data from [link1](https://zenodo.org/records/3625992#.Xiv9jGhKhPY) or [link2](https://mega.nz/#!O6wXlSTS!wcEoDT4Ctq5HRq_hV-aWeVF1_JB3cacQBQqOLjCIbc8). Please refer to [ms-tcn](https://github.com/yabufarha/ms-tcn) or [CVPR2024-FACT](https://github.com/ZijiaLewisLu/CVPR2024-FACT).  
**EgoProceL**: download EgoProceL data from [G-Drive](https://drive.google.com/drive/folders/1qYPLb7Flcl0kZWXFghdEpvrrkTF2SBrH). Please refer to [CVPR2024-FACT](https://github.com/ZijiaLewisLu/CVPR2024-FACT).  
**EgoPER**: download EgoPER data from [G-Drive](https://drive.google.com/drive/folders/1qYPLb7Flcl0kZWXFghdEpvrrkTF2SBrH). Please refer to [EgoPER](https://www.khoury.northeastern.edu/home/eelhami/egoper.htm) for the original data.  

# Usage

## Preprocessing

To generate the target progress values:
```
python utils/write_progress_values.py 
```
To generate task graphs from video transcripts:
```
python utils/write_graph.py 
```

## Training

To train the model, use the following command:

```
python main.py --action train --dataset <dataset_name> --split <split_number> --exp_id protas --causal --graph --learnable_graph [other options]
```

## Testing
To test the model, use the following command:

```
python main.py --action predict --dataset <dataset_name> --split <split_number> --exp_id protas --causal --graph --learnable_graph [other options]
```

**Note:** Theoretically, to test the model in an online setting, you should use the `--action predict_online` argument, which makes predictions frame by frame. However, if the model is set to be causal, it will only make predictions based on frames up to the current frame. In this case, using `--action predict` will produce the same results while being more efficient.


## Citation
If you find the project helpful, we would appreciate if you cite the work:

```
@article{Shen:CVPR24,
  author = {Y.~Shen and E.~Elhamifar},
  title = {Progress-Aware Online Action Segmentation for Egocentric Procedural Task Videos},
  journal = {{IEEE} Conference on Computer Vision and Pattern Recognition},
  year = {2024}}
```


