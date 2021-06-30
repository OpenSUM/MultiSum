for SPLIT in train valid test; do \
    python -m examples.roberta.multiprocessing_bpe_encoder \
        --encoder-json gpt2_bpe/encoder.json \
        --vocab-bpe gpt2_bpe/vocab.bpe \
        --inputs SHT/SHT.${SPLIT}.raw \
        --outputs SHT/SHT.${SPLIT}.bpe \
        --keep-empty \
        --workers 60; \
done