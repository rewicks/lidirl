# Introduction

This is a walkthrough of the whole pipeline---from processing training data to inference. If you're looking for more information on Architectures or Augmentations, please look at the other READMEs.

## Training Data

It is expected that you have both training and development data sorted out. We recommend using the Twitter dataset for training and the FLORES dataset for dev. It is unlikely that there is FLORES data in the Twitter, but you should probably dedup it.

You can pre-apply augmentations offline before training. This may be useful for the development data--to see how well the model is learning augmentations _during_ training, but the training pipeline will apply augmentations _dynamically_ to the training data, so it isn't necessary to do that beforehand.

## Preprocess

The preprocessing script is used to prepare the input data for the training pipeline. Because of the dynamic nature of the augmentations, there isn't a significant amount of preprocessing done to speed up training.

It is assumed that the training data has the format:

```
langid\ttext
```

or

```
en    This is an example text
```

If you don't have data in this format, you can easily create it with awk. Something like:

```
cat dev.en | awk -l lang="en" '{print lang "\t" $0 }' > dev.labels.en
```

Example call:
```
python preprocess.py --train_files data/train* \
                        --valid_files data/dev/* \
                        --smart_group_validation_files \
                        --output_dir preprocessed_data \
                        --temperature 0.3
```

This will process the data into shards (randomly assigned) and put it in a new output directory as a binarized file.

The preprocessing will also perform temperature sampling on the training data--it does this statically and not dynamically before training.

## Training

For more information on the models, see the MODELS.md README.

Visual training, and token_level training will slow down trianing the most.

Full arguments:

```
usage: trainer.py [-h] [--raw_data_dir RAW_DATA_DIR] [--preprocessed_data_dir PREPROCESSED_DATA_DIR] --output_path OUTPUT_PATH [--multilabel] [--token_level] [--checkpoint_path CHECKPOINT_PATH] [--max_tokens MAX_TOKENS]
                  [--tb_dir TB_DIR] [--warmup_lr WARMUP_LR] [--warmup_updates WARMUP_UPDATES] [--lr LR] [--step_rate STEP_RATE] [--embedding_dim EMBEDDING_DIM] [--hidden_dim HIDDEN_DIM] [--dropout DROPOUT] [--cpu]
                  [--min_epochs MIN_EPOCHS] [--max_epochs MAX_EPOCHS] [--save_every_epoch] [--checkpoint_last] [--update_interval UPDATE_INTERVAL] [--log_interval LOG_INTERVAL] [--validation_interval VALIDATION_INTERVAL]
                  [--patience PATIENCE] [--max-updates MAX_UPDATES] [--num-layers NUM_LAYERS] [--montecarlo_layer] [--augmentations AUGMENTATIONS] [--augmentation_probability AUGMENTATION_PROBABILITY]
                  [--wandb_proj WANDB_PROJ] [--wandb_run WANDB_RUN]
                  {linear-ngram,transformer,roformer,convolutional,flash,unet} ...


positional arguments:
  {linear-ngram,transformer,roformer,convolutional,flash,unet}
                        Determines the type of model to be built and trained
    linear-ngram        a linear ngram style model
    transformer         a transformer model
    roformer            a roformer model
    convolutional       a convolutional model
    flash               the new flash model (for testing)
    unet                a unet model

optional arguments:
  -h, --help            show this help message and exit
  --raw_data_dir RAW_DATA_DIR
                        The path to raw data. If this is passed, data directory must have a train and valid subfolder.
  --preprocessed_data_dir PREPROCESSED_DATA_DIR
                        The path to the directory where the data is located. This must be the output of the preprocess call
  --output_path OUTPUT_PATH
                        The path to the directory will output models will be saved
  --multilabel
  --token_level
  --max_tokens MAX_TOKENS
                        The batch size in tokens
  --tb_dir TB_DIR       The directory path to save tensorboard files
  --warmup_lr WARMUP_LR
                        The LR to use during the warmup updates
  --warmup_updates WARMUP_UPDATES
                        The number of updates to warm up (different learning rate)
  --lr LR               The learning rate to use
  --step_rate STEP_RATE
                        The rate at which to decay the learning rate
  --embedding_dim EMBEDDING_DIM
                        The size of the embedding dimension
  --hidden_dim HIDDEN_DIM
                        The size of the hidden dimension.
  --dropout DROPOUT     Dropout percent to use for regularization
  --cpu                 Forces use of cpu even if CUDA is available.
  --min_epochs MIN_EPOCHS
                        Minimum number of epochs to train for.
  --max_epochs MAX_EPOCHS
                        Maximum number of epochs to train for.
  --save_every_epoch    If true, saves a model after each epoch
  --checkpoint_last     If true, saves the last model separate from best model.
  --update_interval UPDATE_INTERVAL
                        Backprops every N updates
  --log_interval LOG_INTERVAL
                        Waits N updates to log information
  --validation_interval VALIDATION_INTERVAL
                        Waits N updates to validate
  --patience PATIENCE   If loss has not improved in N validations, stops training early.
  --max-updates MAX_UPDATES
                        The maximum number of updates before halting training.
  --num-layers NUM_LAYERS
                        The number of layers to use in model.
  --montecarlo_layer    If true, uses a MonteCarlo Layer instead of typical projection layer.
  --augmentations AUGMENTATIONS
                        A comma separated list of augmentation (names) and their ratios.
  --augmentation_probability AUGMENTATION_PROBABILITY
                        The probability of augmenting data.
  --wandb_proj WANDB_PROJ
                        The project where this run will be logged
  --wandb_run WANDB_RUN
                        The name of the run for wandb logging
```

Example run:

```
python trainer.py \
        --preprocessed_data_dir preprocessed_data \
        --output_path out \
        --augmentations "antspeak,1/ngrams,1/hashtag,1/short,1" \
        --augmentation_probability 0.3 \
        roformer
```


## Labeling

The `label.py` can read from standard input or a file.

```
cat input_data |
python lidirl/label.py \
    --model regular/checkpoint_best.pt \
    --complete \
    > labels
```

The `--complete` parameter will give you the entire probability distribution of the model. Alternatively, you can pass `--top` (defaults to 1) which will give you the top-k output of the distribution. The former is a dictionary output and the latter is formatted plain text.

```
usage: label.py [-h] [--model MODEL] [--input INPUT] [--output OUTPUT] [OPTIONS]

LIDIRL: Labels language of input text.
      Example: lidirl --model augment-roformer --input newsdata.txt --output langs.txt

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL, -m MODEL
                        Either name of or path to a pretrained model
  --dataset DATASET
  --input INPUT, -i INPUT
                        Input file. Defaults to stdin.
  --output OUTPUT, -o OUTPUT
                        Path to output file. Defaults to stdout.
  --complete            Stores whether or not to output complete probability distribution. Defaults to False.
  --top TOP
  --multilabel
  --cpu                 Uses CPU (GPU is default if available)
  --batch_size BATCH_SIZE, -b BATCH_SIZE
                        Batch size to use for evaluation.

additional options:
  --version, -V         Prints LIDIRL version
  --download, -D        Downloads model selected via '--model'
  --list, -l            Lists available models.
  --quiet, -q           Disables logging.
  ```