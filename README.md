# BASE
A solution for attribute-aware multi-behavior sequential recommendation.


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
      sample_data.py                        # data preprocessing scripts
      |--data
         |--user_log.csv                 # raw data of Tmall
      |--processed_data          
         |--Tmall_train.csv                 # training data of Tmall        
         |--Tmall_valid.csv                 # valid data of Tmall    
         |--Tmall_between_val_test.csv      # interactions data between the valid item and test item
         |--Tmall_test.csv                  # test data of Tmall    
         |--Tmall_negative.csv              # candidate data of Tmall    
|--NextIP                    
   |--dataset.py                            # data loader
   |--nextip.py                             # model file of our NextIP
   |--evaluate.py                           # evaluation
   |--train.py                               # run file
|--tmall_log.txt                            # the running log of NextIP on the Tmall dataset
```


## Usage

1. Download the datasets and put the files in `../data_process/xx/data/`.

2. Run the data preprocessing scripts to generate the data.
``` 
cd data_process/Tmall
python3 sample_data.py 
```

3. Run the train.py
``` 
cd NextIP
CUDA_VISIBLE_DEVICES=0 python3 train.py 
```

More descriptions of the command arguments are as follws:

| arg_name            | type  | default_value | description                                                         |
|:--------------------|:------|:--------------|:--------------------------------------------------------------------|
| validation          | int   | 1             | Whether to evaluate on the validation set.                          |
| num_epochs          | int   | 500           | Number of epochs.                                                   |
| batch_size          | int   | 128           | Batch size.                                                         |
| lr                  | float | 0.001         | Learning rate.                                                      |
| maxlen              | int   | 50            | Maximum length of sequences.                                        |
| hidden_units        | int   | 50            | latent vector dimensionality.                                       |
| num_blocks          | int   | 1             | Number of self-attention blocks.                                    |
| num_blocks_behavior | int   | 1             | Number of self-attention blocks of behavior-specific item sequence. |
| num_heads           | int   | 1             | Number of heads for attention.                                      |
| dropout_rate        | float | 0.5           | dropout rate.                                                       |
| l2_reg              | float | 0.0           | regularization hyperparameter.                                      |
| eva_interval        | int   | 1             | Number of epoch interval for evaluation.                            |
| wo                  | str   |               | Ablation of behavior-specific item sequence.                        |                              |
