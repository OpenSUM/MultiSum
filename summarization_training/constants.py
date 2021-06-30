PRETRAIN_WORD_EMBEDDING = 1
TRAINABLE_WORD_EMBEDDING = 2
BOTH_WORD_EMBEDDING = 3

CLS_TOKEN = '<s>'
SEP_TOKEN = '</s>'
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'

EXP_DIR = './experiments'

EPSILON = 1e-10

CHECKPOINTS_MAX_TO_KEEP = 999

HALVE_BERT_LR = False
HALVE_OTHER_LR = False
CHECK_GLOBAL_STEPS = 5
PRINT_STEPS = 100
HALVE_LR_STEPS = CHECK_GLOBAL_STEPS * 5
MIN_LEARNING_RATE = 1e-10

SAMPLE_LIMIT = None  
TRAIN_SAMPLE_LIMIT =None
EVAL_SAMPLE_LIMIT = None
MAX_SAMPLE_NUM = None