# MultiSum

**This doc  focuses on introducing  how to using domain word specific method or common method to pretrain roberta base checkpoint.**

**We largely refer to the pre-training framework provided by fairseq,you can find the tutorial [here](https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.pretraining.md).**

### Data preparation For pretraining 

You can download pretraining data [here]().Then run`gpt_2enc.sh` and `preprocess_fairseq.sh` in sequence.

### Start to pretrain

#### Option 1:pretrain in official way provided by fairseq

Set your own parameters and run `train_common.sh`.

#### Option 2: use domain word specific way to pretrain

Replace two files in your fairseq repository with the edition provided by us,including [masked_lm.py](https://github.com/pytorch/fairseq/blob/master/fairseq/tasks/masked_lm.py) and [mask_tokens_dataset.py](https://github.com/pytorch/fairseq/blob/master/fairseq/data/mask_tokens_dataset.py).

Then Set your own parameters and run `train_oov.sh`.