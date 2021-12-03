import os
import argparse
import random
import tempfile
from tqdm import tqdm
import torch
import modeling
import data
import pytrec_eval
from statistics import mean
from collections import defaultdict
import torch.nn as nn
import logging
import pickle
from torch.cuda.amp import GradScaler, autocast


LR = 0.001
BERT_LR = 2e-5
MAX_EPOCH = 10
BATCH_SIZE = 4
BATCHES_PER_EPOCH = 1024

MODEL_MAP = {
    'vanilla_bert': modeling.VanillaBertRanker
    #the other architecture(s) are coming soon
}

#here is the block selection(ranking/scoring) choices
BLOCK_MATCH_MAP = {
    'tf-idf': "tf-idf",
    'bm25': "bm25", # use BM25 to select blocks
    'vanilla': "vanilla", #old truncation, so called vanilla bert
    'bert': "bert", #就是用在训练的模型
    'random' : "random",  #随机选择
}




def get_log(file_name):
     logger = logging.getLogger('train')  # 设定logger的名字
     logger.setLevel(logging.INFO)  # 设定logger得等级
     ch = logging.StreamHandler()  # 输出流的hander，用与设定logger的各种信息
     ch.setLevel(logging.INFO)  # 设定输出hander的level
     fh = logging.FileHandler(file_name, mode='a')  # 文件流的hander，输出得文件名称，以及mode设置为覆盖模式
     fh.setLevel(logging.INFO)  # 设定文件hander得lever
     formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
     ch.setFormatter(formatter)  # 两个hander设置个是，输出得信息包括，时间，信息得等级，以及message
     fh.setFormatter(formatter)
     logger.addHandler(fh)  # 将两个hander添加到我们声明的logger中去
     logger.addHandler(ch)
     return logger

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)

class HingeLoss(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, scores, labels):
        # actually, no need to give labels. The input query-doc pairs, are: 1, 0, 1, 0
        loss = 1.0 - (scores[:, 0] - scores[:, 1])
        loss = torch.clamp(loss, min=0.0)
        loss = torch.mean(loss)
        return loss

scaler = GradScaler()
cuda = torch.cuda.is_available()
def main(model, dataset, train_pairs, qrels_train, valid_run, qrels_valid, logger, args, model_out_dir=None):
    '''
        Runs the training loop, controlled by the constants above
        Args:
            model(torch.nn.model or str): One of the models in modelling.py, 
            or one of the keys of MODEL_MAP.
            dataset: A tuple containing two dictionaries, which contains the 
            text of documents and queries in both training and validation sets:
                ({"q1" : "query text 1"}, {"d1" : "doct text 1"} )
            train_pairs: A dictionary containing query document mappings for the training set
            (i.e, document to to generate pairs from). E.g.:
                {"q1: : ["d1", "d2", "d3"]}
            qrels_train(dict): A dicationary containing training qrels. Scores > 0 are considered
            relevant. Missing scores are considered non-relevant. e.g.:
                {"q1" : {"d1" : 2, "d2" : 0}}
            If you want to generate pairs from qrels, you can pass in same object for qrels_train and train_pairs
            valid_run: Query document mappings for validation set, in same format as train_pairs.
            qrels_valid: A dictionary  containing qrels
            model_out_dir: Location where to write the models. If None, a temporary directoy is used.
    '''

    # if isinstance(model,str):
    #     if cuda:
    #         model = MODEL_MAP[model]().cuda()
    #     else:
    #         model = MODEL_MAP[model]()

    if model_out_dir is None:
        model_out_dir = tempfile.mkdtemp()

    params = [(k, v) for k, v in model.named_parameters() if v.requires_grad]
    non_bert_params = {'params': [v for k, v in params if not k.startswith('bert.')]}
    BERT_LR = args.bert_lr
    bert_params = {'params': [v for k, v in params if k.startswith('bert.')], 'lr': BERT_LR}

    optimizer = torch.optim.Adam([non_bert_params, bert_params], lr=LR)

    total_steps = MAX_EPOCH * args.batch_per_epoch
    warmUpSteps = int(total_steps*0.1)
    # if args.warmup:
    #     scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmUpSteps,
    #                                             num_training_steps=total_steps)
    # else:
    scheduler = []

    epoch = 0
    top_valid_score = None
    PATIENCE = 0
    print(f'Starting training, upto {MAX_EPOCH} epochs, patience {PATIENCE} LR={LR} BERT_LR={BERT_LR}', flush=True)

    criterion = HingeLoss()
    criterion = criterion.cuda() if cuda else criterion

    valid_score = None

    def loadIdfDic(collection_url):
        with open(collection_url + '.pkl', 'rb') as f:
            return pickle.load(f)

    collection_url = args.idf_url

    idf_dic = loadIdfDic(collection_url)
    valid_dic = {}


    for epoch in range(MAX_EPOCH):

        loss = train_iteration(model, optimizer, scheduler, dataset, train_pairs, qrels_train, epoch, criterion, logger, args, idf_dic)

        logger.info('------------------------train epoch=%d loss=%f' %(epoch,loss))
        logger.info('------------------------------------')

        if epoch%1==0:
            valid_score = validate(model, dataset, valid_run, qrels_valid, epoch, args, idf_dic, valid_dic)
            logger.info('validation epoch=%d score=%f' %(epoch,valid_score))

            noParallModel = model.module if hasattr(model, 'module') else model
            noParallModel.save(os.path.join(model_out_dir, str(epoch)+'_weights.p'))

    return (model)

def train_iteration(model, optimizer, scheduler, dataset, train_pairs, qrels, epoch, criterion, logger, args, idf_dic):
    
    total = 0
    model.train()
    total_loss = 0.
    batch_count = 0

    BATCHES_PER_EPOCH = args.batch_per_epoch

    with tqdm('training', total=BATCH_SIZE * BATCHES_PER_EPOCH, ncols=80, desc='train', leave=False,disable=True) as pbar:
        for record in data.iter_train_pairs(model, dataset, train_pairs, qrels, BATCH_SIZE, batch_count, epoch, args, idf_dic):
            if args.amp==False:
                model.train()
                labels = record['label']
                labels = torch.Tensor(labels)
                if cuda:
                    labels = labels.cuda()

                batch_count += 1
                scores = model(record['query_tok'],
                               record['query_mask'],
                               record['doc_tok'],
                               record['doc_mask'])
                count = len(record['query_id'])
                # scores = scores.reshape(count, 2)

                scores = scores.reshape(count // 2, 2)


                loss = criterion(scores, labels)

                loss.backward()
                lossItem = loss.item()
                # total_loss += loss.item()
                total_loss += lossItem
                qidd = record['query_id'][0]
                # tqdm.write(f'qid {qidd}')
                logger.info('qid %s train epoch=%d loss=%f batch_count=%d' % (qidd, epoch, lossItem, batch_count))
                total += count
                # if args.warmup == True:
                #     scheduler.step()
                # 原来的是16，但是原来那个是16个pair，原来那个每次2对，集齐16对，我这个每次4个，集齐32个
                # if total % 32 == 0:
                if args.GA==True:
                    if total % 32 == 0:
                        optimizer.step()
                        if args.warmup==True:
                            scheduler.step()
                        optimizer.zero_grad()

                else:
                    optimizer.step()
                    if args.warmup == True:
                        scheduler.step()
                    optimizer.zero_grad()

                pbar.update(count)
                if total >= BATCH_SIZE * BATCHES_PER_EPOCH:
                    return total_loss
            else:
                with autocast():
                    model.train()
                    labels = record['label']
                    labels = torch.Tensor(labels)
                    if cuda:
                        labels = labels.cuda()

                    batch_count += 1
                    scores = model(record['query_tok'],
                                   record['query_mask'],
                                   record['doc_tok'],
                                   record['doc_mask'])
                    count = len(record['query_id'])
                    # scores = scores.reshape(count, 2)
                    scores = scores.reshape(count // 2, 2)

                    loss = criterion(scores, labels)

                scaler.scale(loss).backward()
                lossItem = loss.item()
                # total_loss += loss.item()
                total_loss += lossItem
                qidd = record['query_id'][0]
                # tqdm.write(f'qid {qidd}')
                logger.info('qid %s train epoch=%d loss=%f batch_count=%d' % (qidd, epoch, lossItem, batch_count))
                total += count
                # if args.warmup == True:
                #     scheduler.step()
                # 原来的是16，但是原来那个是16个pair，原来那个每次2对，集齐16对，我这个每次4个，集齐32个
                # if total % 32 == 0:
                if args.GA == True:
                    if total % 32 == 0:
                        # optimizer.step()
                        scaler.step(optimizer)
                        scaler.update()
                        if args.warmup == True:
                            scheduler.step()
                        optimizer.zero_grad()

                else:
                    # optimizer.step()
                    scaler.step(optimizer)
                    scaler.update()

                    if args.warmup == True:
                        scheduler.step()
                    optimizer.zero_grad()

                pbar.update(count)
                if total >= BATCH_SIZE * BATCHES_PER_EPOCH:
                    return total_loss



def validate(model, dataset, run, valid_qrels, epoch, args, idf_dic, valid_dic):
    run_scores = run_model(model, dataset, run, args, idf_dic)

    metrics = ['ndcg_cut.1', 'ndcg_cut.3', 'ndcg_cut.5', 'ndcg_cut.10', 'ndcg_cut.20', 'ndcg', 'P.1', 'P.3', 'P.5', 'P.10', 'P.20', 'map']
    trec_eval = pytrec_eval.RelevanceEvaluator(valid_qrels, metrics)
    eval_scores = trec_eval.evaluate(run_scores)
    keys = ['map', 'P_1', 'P_3', 'P_5', 'P_10', 'P_20', 'ndcg', 'ndcg_cut_1', 'ndcg_cut_3', 'ndcg_cut_5', 'ndcg_cut_10', 'ndcg_cut_20']
    for item in keys:
        meanItemRes = mean([d[item] for d in eval_scores.values()])
        valid_dic.setdefault(epoch, {})[item] = meanItemRes
    print(eval_scores)
    return mean([d['ndcg_cut_20'] for d in eval_scores.values()])


def run_model(model, dataset, run, args, idf_dic, desc='valid'):
    rerank_run = defaultdict(dict)
    runBatch = BATCH_SIZE
    with torch.no_grad(), tqdm(total=sum(len(r) for r in run.values()), ncols=80, desc=desc, leave=False) as pbar:
        model.eval()
        for records in data.iter_valid_records(model, dataset, run, runBatch, args, idf_dic):
            if args.amp==True:
                with autocast():
                    scores = model(records['query_tok'],
                                   records['query_mask'],
                                   records['doc_tok'],
                                   records['doc_mask'])
            else:
                scores = model(records['query_tok'],
                               records['query_mask'],
                               records['doc_tok'],
                               records['doc_mask'])

            for qid, did, score in zip(records['query_id'], records['doc_id'], scores):
                rerank_run[qid][did] = score.item()
            pbar.update(len(records['query_id']))
    return rerank_run
    

def write_run(rerank_run, runf):
    '''
        Utility method to write a file to disk. Now unused
    '''
    with open(runf, 'wt') as runfile:
        for qid in rerank_run:
            scores = list(sorted(rerank_run[qid].items(), key=lambda x: (x[1], x[0]), reverse=True))
            for i, (did, score) in enumerate(scores):
                runfile.write(f'{qid} 0 {did} {i+1} {score} run\n')

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def main_cli():
    parser = argparse.ArgumentParser('CEDR model training and validation')
    parser.add_argument('--model', choices=MODEL_MAP.keys(), default='vanilla_bert')
    parser.add_argument('--bert_num', type=int, default=1)
    parser.add_argument('--batch_per_epoch', type=int, default=1024)
    parser.add_argument('--block_match', choices=BLOCK_MATCH_MAP.keys(), default='vanilla')

    parser.add_argument('--datafiles', type=argparse.FileType('rt'), nargs='+')
    parser.add_argument('--qrels', type=argparse.FileType('rt'))
    parser.add_argument('--train_pairs', type=argparse.FileType('rt'))
    parser.add_argument('--valid_run', type=argparse.FileType('rt'))
    parser.add_argument('--initial_bert_weights', type=argparse.FileType('rb'))
    parser.add_argument('--model_out_dir')
    parser.add_argument('--bert_lr', type=float)
    parser.add_argument('--GA', action='store_true')
    # parser.add_argument('--warmup', action='store_true')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--idf_url') #IDF dictionary URL
    args = parser.parse_args()

    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)

    # model = MODEL_MAP[args.model]().cuda()
    cuda = torch.cuda.is_available()

    if cuda:
        model = MODEL_MAP[args.model]().cuda()
    else:
        model = MODEL_MAP[args.model]()

    mkdir(args.model_out_dir)
    logger = get_log(os.path.join(args.model_out_dir, 'log.txt'))

    dataset = data.read_datafiles(args.datafiles)
    qrels = data.read_qrels_dict(args.qrels)
    train_pairs = data.read_pairs_dict(args.train_pairs)

    valid_run = data.read_run_dict(args.valid_run)


    if args.initial_bert_weights is not None:
        model.load(args.initial_bert_weights.name)
    os.makedirs(args.model_out_dir, exist_ok=True)
    # we use the same qrels object for both training and validation sets
    main(model, dataset, train_pairs, qrels, valid_run, qrels, logger, args, args.model_out_dir)


if __name__ == '__main__':
    main_cli()
