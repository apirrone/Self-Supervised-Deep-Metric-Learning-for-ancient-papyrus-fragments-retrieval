# Code for paper "Self-Supervised Deep Metric Learning for ancient papyrus fragments retrieval

cite as : 

```
@article{pirrone2021self,
  title={Self-supervised deep metric learning for ancient papyrus fragments retrieval},
  author={Pirrone, Antoine and Beurton-Aimar, Marie and Journet, Nicholas},
  journal={International Journal on Document Analysis and Recognition (IJDAR)},
  pages={1--16},
  year={2021},
  publisher={Springer}
}
```

## Documentation

Note, all the scripts have a `-h` for help on their arguments.

### First generate a dataset

In `make_dataset.py`, check the dataset parameters:

- `nb_papyri` : number of papyri to use in the dataset
- `patch_size` : size of the patches in pixels **(64**)
- `data_path` : path to the directory containing the full images (of fragments)
- `test_proportion` : proportion of papyri that will be put aside in the test set (for example, with `nb_papyri` = 100 and `test_proportion`= 0.1, 10 papyri will go in the test set) (**0.1**)
- `max_patches_per_fragments` : maximum number of patches that will be extracted for each fragment image (**5**)
- `fragmentAsClass` : if set to **True**, the training and validation set will be prepared in *self_supervised_learning* mode. If set to **False**, the ground truth will be used to prepare the training and validation sets (*normal* mode). The Test set is always in *normal* mode, so if no ground truth is available, `test_proportion` should be set to **0**
- `resize` : if set, resizes all the images by this value. (**None**)



Depending on the name formatting of your own files, you should change the content of the function `get_papyrus_id()` such that given an image name, the class id is returned (when the `fragmentAsClass` parameter is set to False)

Once all the parameters are correctly set, run : 

```bash
$ python3 make_dataset.py -d <datasetDir> -n <datasetName>
```

### Then start training

In `train.py`, check the training hyperparameters, then run :

````bash
$ python3 train.py -g <gpu_id> -da <path_to_dataset> -d <save_directory>
````

#### Using Tensorboard to monitor training

```bash
$ tensorboard --logdir="<path_to_logs_directory>"
```



### Finally, evaluate the model

Check the evaluation hyperparameters, then run : 

```bash
$ python3 evaluate.py -g <gpu_id> -sa <save_directory> -m <model_directory> -da <path_to_dataset> -e <experiment_name> -s <epoch_step>
```

Depending on the number of papyri in the test set, this can take a while





