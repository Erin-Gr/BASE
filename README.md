# BASE
A solution for attribute-aware multi-behavior sequential recommendation (A-MBSR).


# Environment
Our code is based on the following packages:
- GPU: Tesla V100-PCIE-16GB
- Requirments： 
   - Python = 3.6.9
   - TensorFlow 1.15.3
   - pandas 0.25.3
   - numpy 1.17.3


# Code Structure
```
|--data_process
   |--Tmall
      |--processed_data_att.zip               # experimented dataset (Tmall)
      |--sample_data_att.py                   # data preprocessing scripts
   |--kaggle          
      |--processed_data_att.zip               # experimented dataset (Cosmetics)
      |--sample_data_att.py                   # data preprocessing scripts
   |--JD          
      |--processed_data_att.zip               # experimented dataset (JD)
      |--sample_data_att.py                   # data preprocessing scripts   
|--code                    
   |--attMBSASRec.py                          # main model file of our BASE
   |--att_dataset.py                          # data loader
   |--bslayer.py                              # model sub-module file of our BASE
   |--bsmmoe.py                               # model sub-module file of our BASE
   |--evaluate.py                             # evaluation
   |--train.py                                # run file
|--README.md                                  # readme file
```


# Usage
Run the train.py
``` 
cd code
CUDA_VISIBLE_DEVICES=0 python3 train.py 
```

