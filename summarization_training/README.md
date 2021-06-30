# MultiSum

**This doc  focuses on introducing  how to train and test our summarization models.**

### Data preparation For Amazon Review

#### Option 1:download the processed data

Download and unzip the data from [here]()

#### Option 2:process the data yourself

**Step 1:Download Amazon data**

Download the data from [here](http://jmcauley.ucsd.edu/data/amazon/)

**Step 2:Sentence Splitting and Tokenization**

transform downloaded datasets into train/eval/predict files

run cut_json.py transform json files into format we want

run tokenize_word.py  Filter out cases whose length does not meet the requirements and tokenize sentences

run train_eval_predict_split.py

### Start to train

run run.py with given parameters,value of parameters can be set in bert_config.json and in command lines.

You can initialize model parameters from different checkpoints(e.g.:roberta base ,pretrained...) by using    `--init_checkpoint --roberta_checkpoint --pretrained_checkpoint`.

If you choose to use roberta base as your initializing checkpoint,you can download it [here]()

e.g.:

```
python -u run.py \
	--num_gpus=4 \
	--mode=train \
	--batch_size=64 \
	--bert_learning_rate=5e-5 \
	--use_pointer=true \
	--coverage=true \
```

### Test the result of given checkpoint

run run.py in test mode with given checkpoint path

e.g.:

```
python -u run.py \ 
         --mode=test \
         --init_checkpoint=./experiments/my_checkpoint \
         --checkpoint_file=best-1 \
         --num_gpus=4 \
         --use_pointer=true \
         --coverage=true \
```



### How to transform pytorch checkpoint into tensorflow:

Facebook only issue Roberta in pytorch edition.Moreover,The pre-training framework can only generate checkpoint files of .pt files.However,our framework can only initalize parameters by using tensorflow edition checkpoint.So you have to transform .pt checkpoint into tensorflow checkpoint.

We refer the code in this [repository](https://github.com/vickyzayats/roberta_tf_ckpt) and [code from  transformers repository](https://github.com/huggingface/transformers/blob/master/src/transformers/models/roberta/convert_roberta_original_pytorch_checkpoint_to_pytorch.py) . The whole process can be finished by running codes in folder named transform_pytorch_to_tensorflow.

e.g.:

```
python convert_roberta_original_pytorch_checkpoint_to_pytorch.py 
--roberta_checkpoint_path=roberta_base_from_fairseq 
--pytorch_dump_folder_path=your_dump_path
```

then run:

```
python convert_pytorch_checkpoint_to_tf.py 
--model_name=your_dump_path/pytorch_model.bin
--config_file=roberta_config.json --tf_cache_dir=./tf_checkpoint
```

