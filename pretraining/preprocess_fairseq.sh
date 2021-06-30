fairseq-preprocess \
    --only-source \
    --srcdict gpt2_bpe/dict.txt \
    --trainpref SHT/SHT.train.bpe \
    --validpref SHT/SHT.valid.bpe \
    --testpref SHT/SHT.test.bpe \
    --destdir SHT-data-bin/SHT \
    --workers 60