## PAT-Net (Part-independent Attention Network for Skeleton Based Human Action Recognition)
 The PAT-Net contains a whitened pairwise self-attention, unary self-attention and position attention as independent functions and different projection matrices for learning representative action features.  The whitened pairwise self-attention captures the influence of a single key joint specifically on another query joint, and the unary self-attention models the general impact of one key joint over all other query joints to learn the discriminative action features. Furthermore, we design a position attention learning module that computes the correlation between  action semantics and position information separately with different projection matrices. 



## Python >= 3.6

- PyTorch >= 1.1.0
- PyYAML, tqdm, tensorboardX

# Data Preparation
Download datasets.
There are 3 datasets to download:

- NTU RGB+D 60 Skeleton
- NTU RGB+D 120 Skeleton
- UESTC skeleton Datatset

NTU RGB+D 60 and 120 Dataset
Request dataset here: http://rose1.ntu.edu.sg/Datasets/actionRecognition.asp
Download the skeleton datasets:
- nturgbd_skeletons_s001_to_s017.zip (NTU RGB+D 60)
- nturgbd_skeletons_s018_to_s032.zip (NTU RGB+D 120)
- Extract above files to ./data/nturgbd_raw
- 
UESTC Dataset
Request dataset here:  https://github.com/HRI-UESTC/CFM-HRI-RGB-D-action-database

## Dataset Preparation.  
Put downloaded data into the following directory structure:

- data/
  - UESTC/
      ... # raw data of UESTC
  - ntu/
  - ntu120/
  - nturgbd_raw/
    - nturgb+d_skeletons/     # from `nturgbd_skeletons_s001_to_s017.zip`
      ...
    - nturgb+d_skeletons120/  # from `nturgbd_skeletons_s018_to_s032.zip`
 
Generating Data:
```
cd ./data/ntu # or cd ./data/ntu120
 # Get skeleton of each performer
 python get_raw_skes_data.py
 # Remove the bad skeleton 
 python get_raw_denoised_data.py
 # Transform the skeleton to the center of the first frame
 python seq_transformation.py
```

## Training and Testing 
Training
- Change the config file depending on what you want.
```
# Example: training PAT-Net on NTU RGB+D 60 cross subject
 python main.py --config ./config/nturgbd-cross-subject/joint.yaml
 python main.py --config ./config/nturgbd-cross-subject/bone.yaml
 python main.py --config ./config/nturgbd-cross-subject/joint_motion.yaml
 python main.py --config ./config/nturgbd-cross-subject/bone_motion.yaml
```
Testing
- To test the trained models saved in <work_dir>, run the following command:
```
python main.py --config <work_dir>/config.yaml --work-dir <work_dir> --phase test --save-score True --weights <work_dir>/weight.pt

```
- To ensemble the results of different modalities, run
 ```
 python ensemble.py 
 ```
 ## Acknowledgements
 This repo is based on [2s-AGCN](https://github.com/lshiwjx/2s-AGCN) and [ST-TR](https://github.com/). The data processing is borrowed from [SGN](https://github.com/microsoft/SGN)

Thanks to the original authors for their work!

# Contact
For any questions, feel free to contact: `alemugedamu@gmail.com`
