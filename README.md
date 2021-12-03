# keyB

The model is implemented using PyTorch.

##### clone or download the repo/

git clone ....

##### Prepare data

Please see https://github.com/Georgetown-IR-Lab/cedr , the data format

For example, prepare MQ2007 data, according to the format.

Due to the dataset's license, they are unable to be shared...

Then generate IDF dictionary (python dictionary, stored using pickle, all words are lower cased)



##### Train a model:

For example, after preparing MQ2007 data, when using BM25 to select blocks, can:

```shell
python train.py
--model vanilla_bert \
--batch_per_epoch 2048 \
--block_match bm25 \
--datafiles /path to data/queries.tsv /path to data/documents.tsv \
--qrels ${qrels_path} \
--train_pairs /path to data/f1.train.pairs \
--valid_run /path to data/f1.vali.pairs \
--model_out_dir /save model dir \
--bert_lr 2e-5 \
--GA \
--fold 1 \
--seed ${seed} \
--idf_url /the IDF url
--amp
```

When using BERT to select blocks, that is to say, KeyB(vBERT)_BinB, can change the --block_match to 'bert'



After training, could load a model and use the run_model function to get run scores.



The repository is still being sorted up/fulfilled/improved, and would be finished within or soon after the long paper is accepted by a journal etc.

### Citation

If you use this work, please cite:

```
@inproceedings{li2021keybld,
  title={KeyBLD: Selecting Key Blocks with Local Pre-ranking for Long Document Information Retrieval},
  author={Li, Minghan and Gaussier, Eric},
  booktitle={Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={2207--2211},
  year={2021}
}
```

```
@article{li2021power,
  title={The Power of Selecting Key Blocks with Local Pre-ranking for Long Document Information Retrieval},
  author={Li, Minghan and Popa, Diana Nicoleta and Chagnon, Johan and Cinar, Yagmur Gizem and Gaussier, Eric},
  journal={arXiv preprint arXiv:2111.09852},
  year={2021}
}
```



