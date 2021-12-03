import random
from tqdm import tqdm
import torch
import numpy as np
import math
from genePassage import Passage
from torch.cuda.amp import GradScaler, autocast

def read_datafiles(files):
    queries = {}
    docs = {}
    for file in files:
        for line in tqdm(file, desc='loading datafile (by line)', leave=False):
            cols = line.rstrip().split('\t')
            if len(cols) != 3:
                tqdm.write(f'skipping line: `{line.rstrip()}`')
                continue
            c_type, c_id, c_text = cols
            assert c_type in ('query', 'doc')
            if c_type == 'query':
                queries[c_id] = c_text
            if c_type == 'doc':
                docs[c_id] = c_text
    return queries, docs


def read_qrels_dict(file):
    result = {}
    for line in tqdm(file, desc='loading qrels (by line)', leave=False):
        qid, _, docid, score = line.split()
        result.setdefault(qid, {})[docid] = int(score)
    return result


def read_run_dict(file):
    result = {}
    for line in tqdm(file, desc='loading run (by line)', leave=False):
        qid, _, docid, rank, score, _ = line.split()
        result.setdefault(qid, {})[docid] = float(score)
    return result


def read_pairs_dict(file):
    result = {}
    for line in tqdm(file, desc='loading pairs (by line)', leave=False):
        qid, docid = line.split()
        result.setdefault(qid, {})[docid] = 1
    return result

def commonText(str1, str2):
    str1 = str1.lower()
    str2 = str2.lower()
    str1Arr = str1.split(" ")
    str2Arr = str2.split(" ")
    return list(set(str1Arr).intersection(set(str2Arr)))

def calBm25(commonWordList, passage, blockAvgLength, idf_dic, k1, b, stemming=False):
    if len(commonWordList)==0:
        return 0
    # if stemming:
    #     passage = strStem(passage)
    score = 0
    for word in commonWordList:
        idf_val = idf_dic.get(word)
        if idf_val is None:
            idf_val = 10.0  # 未看见的
        #求tf，term frequency,该word出现次数
        tf = 0
        passageStrList = passage.split(" ")
        for passage_word in passageStrList:
            if passage_word==word:
                tf += 1
        #begin to calculate bm25
        passageLength = len(passageStrList)
        ld_div = 1
        if blockAvgLength != 0:
            ld_div = passageLength/blockAvgLength #should be avg length
        numerator = tf
        denumerator = k1*(1-b+b*ld_div)+tf
        relev = numerator/denumerator
        bm25 = idf_val*relev
        score += bm25
    return score


def caltf_idf(commonWordList, passage, idf_dic, stemming=False):
    if len(commonWordList)==0:
        return 0
    # if stemming:
    #     passage = strStem(passage)
    score = 0
    for word in commonWordList:
        idf_val = idf_dic.get(word)
        if idf_val is None:
            idf_val = 10.0  # 未看见的，给个默认
        #求tf，term frequency,该word出现次数
        tf = 0
        for passage_word in passage.split(" "):
            if passage_word==word:
                tf += 1
        if tf>0:
            tf = math.log10(tf) + 1
        else:
            tf = 0.0
        tf_idf= tf*idf_val
        score += tf_idf
    return score

def calWholeDocAndSample(passages, commonwords, stemming, query_length, idf_dic, bert_num):
    for block in passages.blocks:
        blockStr = str(block).lower()
        block.relevance = caltf_idf(commonwords, blockStr, idf_dic, stemming)
    # 获得删除了冗余之后的文档
    return generateDocByScores(passages.blocks, query_length, bert_num)

#使用正在训练的那个bert评分
def calWholeDocAndSampleBert(passages, model, query_tok, query_length, bert_num, args):
    n = 16 #batch大小
    model.eval()
    for batch_blocks in [passages.blocks[i:i + n] for i in range(0, len(passages.blocks), n)]:
        # batch = {'query_tok': [], 'doc_tok': []}
        batch = {'query_id': [], 'doc_id': [], 'query_tok': [], 'doc_tok': [], 'label': []}
        for block in batch_blocks:
            batch['query_tok'].append(query_tok)
            # batch['doc_tok'].append(block.tokens)
            batch['doc_tok'].append(model.tokenize(str(block)))
        # batch['doc_tok'] = list(map(lambda x: x.tokens, batch_blocks))
        # batch['doc_tok'] = list(map(lambda x: tokenModel.tokenize(str(x)), batch_blocks))
        record = _pack_n_ship(batch, bert_num)
        if args.amp==True:
            with autocast():
                with torch.no_grad():
                    scores = model(record['query_tok'],
                                   record['query_mask'],
                                   record['doc_tok'],
                                   record['doc_mask'])
        else:
            with torch.no_grad():
                scores = model(record['query_tok'],
                               record['query_mask'],
                               record['doc_tok'],
                               record['doc_mask'])
        for i in range(len(scores)):
            score = scores[i].item()
            block = batch_blocks[i]
            block.relevance = score

    # 获得删除了冗余之后的文档
    return generateDocByScores(passages.blocks, query_length, bert_num)


def calWholeDocAndSampleBm25(passages, commonwords, stemming, query_length, idf_dic, bert_num, k1=0.9, b=0.4):
    blockLenList = []
    for block in passages.blocks:
        blockStr = str(block).lower()
        # if stemming:
        #     blockStr = strStem(blockStr)
        blockStrLen = len(blockStr.split(" "))
        blockLenList.append(blockStrLen)
    # blockAllLength = np.sum(blockLenList)
    blockAvgLength = np.mean(blockLenList)
    for block in passages.blocks:
        blockStr = str(block).lower()
        block.relevance = calBm25(commonwords, blockStr, blockAvgLength, idf_dic, k1, b, stemming)
        # block.relevance = caltf_idf(commonwords, blockStr, idf_dic, stemming)
    # 获得删除了冗余之后的文档
    return generateDocByScores(passages.blocks, query_length, bert_num)

#随机选择，实际就是随机打分，然后选择哈。。。
def calWholeDocAndSampleRandomly(passages, commonwords, stemming, query_length, idf_dic, bert_num):
    for block in passages.blocks:
        block.relevance = random.random()
    # 获得删除了冗余之后的文档
    return generateDocByScores(passages.blocks, query_length, bert_num)

#根据bert容量和block的相关性分数，选择block生成document
def generateDocByScores(blocks, query_length, bert_num):
    sort_blocks = sorted(blocks, key=lambda x: x.relevance, reverse=True) #according to relevance descending
    candidates = []
    sum_len = 0
    for block in sort_blocks:
        if sum_len >= (512-3-query_length)*bert_num:
            break
        # if (sum_len+len(block))>=(800-3-query_length):
        #     break
        candidates.append(block)
        sum_len += len(block)
    #then re order by pos
    sort_blocks2 = sorted(candidates, key=lambda x: x.pos, reverse=False) #according to relevance descending
    docList = []
    for bl in sort_blocks2:
        docList.append(str(bl))
    document = " ".join(docList)
    return document


#ori是用原来的截断，ori-false是用tf-idf等
def iter_train_pairs(model, dataset, train_pairs, qrels, batch_size, batch_count, epoch, args, idf_dic):
    batch = {'query_id': [], 'doc_id': [], 'query_tok': [], 'doc_tok': [], 'label': []}
    bert_num = args.bert_num
    useSeed = (True if (args.collection=='robust04') else False)

    if args.block_match=='vanilla':
        for qid, did, query_tok, doc_tok, label in _iter_train_pairs_ori(model, dataset, train_pairs, qrels, batch_count,
                                                                     epoch,useSeed):
            batch['query_id'].append(qid)
            batch['doc_id'].append(did)
            batch['query_tok'].append(query_tok)
            batch['doc_tok'].append(doc_tok)
            batch['label'].append(label)
            if len(batch['query_id']) == batch_size:
                yield _pack_n_ship(batch, bert_num)
                # model = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
                # model = model.cuda()
                batch = {'query_id': [], 'doc_id': [], 'query_tok': [], 'doc_tok': [], 'label': []}
    else:
        for qid, did, query_tok, doc_tok, label in _iter_train_pairs(model, dataset, train_pairs, qrels, batch_count,
                                                                     epoch,args,idf_dic,useSeed):
            batch['query_id'].append(qid)
            batch['doc_id'].append(did)
            batch['query_tok'].append(query_tok)
            batch['doc_tok'].append(doc_tok)
            batch['label'].append(label)
            if len(batch['query_id']) == batch_size:
                yield _pack_n_ship(batch, bert_num)
                # model = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
                # model = model.cuda()
                batch = {'query_id': [], 'doc_id': [], 'query_tok': [], 'doc_tok': [], 'label': []}


# from transformers import BertTokenizer
# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
def _iter_train_pairs_ori(model, dataset, train_pairs, qrels, batch_count, epoch,useSeed):
    ds_queries, ds_docs = dataset
    while True:
        qids = list(train_pairs.keys())
        random.shuffle(qids)
        for qid in qids:
            pos_ids = [did for did in train_pairs[qid] if qrels.get(qid, {}).get(did, 0) > 0]
            if len(pos_ids) == 0:
                tqdm.write("no positive labels for query %s " % qid)
                continue
            pos_ids_lookup = set(pos_ids)
            # pos_ids_set = set(pos_ids)
            neg_ids = [did for did in train_pairs[qid] if did not in pos_ids_lookup]
            if len(neg_ids) == 0:
                tqdm.write("no negative labels for query %s " % qid)
                continue
            pos_id_array = []
            neg_id_array = []
            pos_doc_array = []
            neg_doc_array = []
            label_array = []
            for i in range(2):
                pos_id_i = random.choice(pos_ids)
                pos_label =  qrels.get(qid, {}).get(pos_id_i, 0)
                label_array.append(pos_label) # rel=1 or 2
                label_array.append(0)
                neg_id_i = random.choice(neg_ids)
                pos_doc_i = ds_docs.get(pos_id_i)
                if pos_doc_i is None:
                    tqdm.write(f'missing doc {pos_id_i}! Skipping')
                    continue
                neg_doc_i = ds_docs.get(neg_id_i)
                if neg_doc_i is None:
                    tqdm.write(f'missing doc {neg_id_i}! Skipping')
                    continue
                pos_id_array.append(pos_id_i)
                neg_id_array.append(neg_id_i)
                pos_doc_array.append(pos_doc_i)
                neg_doc_array.append(neg_doc_i)

            tokenModel = model.module if hasattr(model, 'module') else model

            query_tok = tokenModel.tokenize(ds_queries[qid])

            # pos_doc = ds_docs.get(pos_id)

            if len(pos_id_array)<2:
                tqdm.write(f'array length {qid} < 6')
                continue

            yield qid, pos_id_array[0], query_tok, tokenModel.tokenize(pos_doc_array[0]), label_array[0]
            yield qid, neg_id_array[0], query_tok, tokenModel.tokenize(neg_doc_array[0]), label_array[1]

            yield qid, pos_id_array[1], query_tok, tokenModel.tokenize(pos_doc_array[1]), label_array[2]
            yield qid, neg_id_array[1], query_tok, tokenModel.tokenize(neg_doc_array[1]), label_array[3]

#经过了tf-idf的
def _iter_train_pairs(model, dataset, train_pairs, qrels, batch_count, epoch, args, idf_dic, useSeed):
    stemming = False  # 是否stemming
    ds_queries, ds_docs = dataset
    while True:
        qids = list(train_pairs.keys())

        random.shuffle(qids)
        for qid in qids:
            pos_ids = [did for did in train_pairs[qid] if qrels.get(qid, {}).get(did, 0) > 0]
            if len(pos_ids) == 0:
                tqdm.write("no positive labels for query %s " % qid)
                continue
            pos_ids_lookup = set(pos_ids)
            # pos_ids_set = set(pos_ids)
            neg_ids = [did for did in train_pairs[qid] if did not in pos_ids_lookup]
            if len(neg_ids) == 0:
                tqdm.write("no negative labels for query %s " % qid)
                continue
            pos_id_array = []
            neg_id_array = []
            pos_doc_array = []
            neg_doc_array = []
            label_array = []
            tokenModel = model.module if hasattr(model, 'module') else model
            query_tok = tokenModel.tokenize(ds_queries[qid])
            query_length = len(query_tok)

            for i in range(2):

                pos_id_i = random.choice(pos_ids)
                pos_label =  qrels.get(qid, {}).get(pos_id_i, 0)
                label_array.append(pos_label) # rel=1 or 2
                label_array.append(0)
                neg_id_i = random.choice(neg_ids)
                pos_doc_i = ds_docs.get(pos_id_i)
                if pos_doc_i is None:
                    tqdm.write(f'missing doc {pos_id_i}! Skipping')
                    continue
                neg_doc_i = ds_docs.get(neg_id_i)
                if neg_doc_i is None:
                    tqdm.write(f'missing doc {neg_id_i}! Skipping')
                    continue
                pos_id_array.append(pos_id_i)
                neg_id_array.append(neg_id_i)

                # processDoc
                queryyyy = ds_queries[qid]
                queryyyy = queryyyy.lower()

                pos_doc_i_passages, cnt = Passage.split_document_into_blocks(tokenModel.tokenizer.tokenize(pos_doc_i),
                                                                             tokenModel.tokenizer, 0, hard=False)
                neg_doc_i_passages, cnt = Passage.split_document_into_blocks(tokenModel.tokenizer.tokenize(neg_doc_i),
                                                                             tokenModel.tokenizer, 0, hard=False)
                # # #如果stem；比如去除复数等
                # if stemming:
                #     queryyyy = strStem(queryyyy)
                #     pos_doc_i = strStem(pos_doc_i)
                #     neg_doc_i = strStem(neg_doc_i)

                bert_num = args.bert_num
                block_match = args.block_match
                commonWordListPos = commonText(queryyyy, pos_doc_i)
                if block_match=='tf-idf':
                    pos_doc_i = calWholeDocAndSample(pos_doc_i_passages, commonWordListPos, stemming, query_length, idf_dic, bert_num)
                elif block_match=='bm25':
                    pos_doc_i = calWholeDocAndSampleBm25(pos_doc_i_passages, commonWordListPos, stemming, query_length, idf_dic, bert_num)
                elif block_match=='bert':
                    pos_doc_i = calWholeDocAndSampleBert(pos_doc_i_passages, tokenModel, query_tok, query_length, bert_num, args)
                elif block_match=="random":
                    pos_doc_i = calWholeDocAndSampleRandomly(pos_doc_i_passages, commonWordListPos, stemming, query_length, idf_dic, bert_num)

                commonWordListNeg = commonText(queryyyy, neg_doc_i)
                if block_match=='tf-idf':
                    neg_doc_i = calWholeDocAndSample(neg_doc_i_passages, commonWordListNeg, stemming, query_length, idf_dic, bert_num)
                elif block_match=='bm25':
                    neg_doc_i = calWholeDocAndSampleBm25(neg_doc_i_passages, commonWordListNeg, stemming, query_length, idf_dic, bert_num)
                elif block_match == 'bert':
                    neg_doc_i = calWholeDocAndSampleBert(neg_doc_i_passages, tokenModel, query_tok, query_length,
                                                         bert_num, args)
                elif block_match=="random":
                    neg_doc_i = calWholeDocAndSampleRandomly(neg_doc_i_passages, commonWordListNeg, stemming, query_length, idf_dic, bert_num)

                pos_doc_array.append(pos_doc_i)
                neg_doc_array.append(neg_doc_i)
            # pos_doc = ds_docs.get(pos_id)

            if len(pos_id_array)<2:
                tqdm.write(f'array length {qid} < 6')
                continue

            yield qid, pos_id_array[0], query_tok, tokenModel.tokenize(pos_doc_array[0]), label_array[0]
            yield qid, neg_id_array[0], query_tok, tokenModel.tokenize(neg_doc_array[0]), label_array[1]

            yield qid, pos_id_array[1], query_tok, tokenModel.tokenize(pos_doc_array[1]), label_array[2]
            yield qid, neg_id_array[1], query_tok, tokenModel.tokenize(neg_doc_array[1]), label_array[3]


def iter_valid_records(model, dataset, run, batch_size, args, idf_dic):
    batch = {'query_id': [], 'doc_id': [], 'query_tok': [], 'doc_tok': [], 'label': []}
    bert_num = args.bert_num
    if args.block_match == 'vanilla':
        for qid, did, query_tok, doc_tok in _iter_valid_records_ori(model, dataset, run):
            batch['query_id'].append(qid)
            batch['doc_id'].append(did)
            batch['query_tok'].append(query_tok)
            batch['doc_tok'].append(doc_tok)
            batch['label'].append(0)
            if len(batch['query_id']) == batch_size:
                yield _pack_n_ship(batch, bert_num)
                batch = {'query_id': [], 'doc_id': [], 'query_tok': [], 'doc_tok': [], 'label': []}
        # final batch
        if len(batch['query_id']) > 0:
            yield _pack_n_ship(batch, bert_num)
    else:
        for qid, did, query_tok, doc_tok in _iter_valid_records(model, dataset, run, args, idf_dic):
            batch['query_id'].append(qid)
            batch['doc_id'].append(did)
            batch['query_tok'].append(query_tok)
            batch['doc_tok'].append(doc_tok)
            batch['label'].append(0)
            if len(batch['query_id']) == batch_size:
                yield _pack_n_ship(batch, bert_num)
                batch = {'query_id': [], 'doc_id': [], 'query_tok': [], 'doc_tok': [], 'label': []}
        # final batch
        if len(batch['query_id']) > 0:
            yield _pack_n_ship(batch, bert_num)


def _iter_valid_records_ori(model, dataset, run):
    ds_queries, ds_docs = dataset
    for qid in run:
        query_tok = model.tokenize(ds_queries[qid])
        for did in run[qid]:
            doc = ds_docs.get(did)
            if doc is None:
                tqdm.write(f'missing doc {did}! Skipping')
                continue
            doc_tok = model.tokenize(doc)
            yield qid, did, query_tok, doc_tok

#下面是经过了tf-idf的
def _iter_valid_records(model, dataset, run,args,idf_dic):
    ds_queries, ds_docs = dataset
    stemming = False
    for qid in run:
        tokenModel = model.module if hasattr(model, 'module') else model

        query_tok = tokenModel.tokenize(ds_queries[qid])
        query_length = len(query_tok)
        for did in run[qid]:
            doc = ds_docs.get(did)
            if doc is None:
                tqdm.write(f'missing doc {did}! Skipping')
                continue

            # processDoc
            queryyyy = ds_queries[qid]
            queryyyy = queryyyy.lower()
            pos_doc_i_passages, cnt = Passage.split_document_into_blocks(tokenModel.tokenizer.tokenize(doc),
                                                                             tokenModel.tokenizer, 0, hard=False)
            # # #如果stem；比如去除复数等
            # if stemming:
            #     queryyyy = strStem(queryyyy)
            #     pos_doc_i = strStem(doc)
            #     # neg_doc_i = strStem(neg_doc_i)

            commonWordListPos = commonText(queryyyy, doc)
            bert_num = args.bert_num
            block_match = args.block_match
            if block_match == 'tf-idf':
                doc = calWholeDocAndSample(pos_doc_i_passages, commonWordListPos, stemming, query_length, idf_dic, bert_num)
            elif block_match == 'bm25':
                doc = calWholeDocAndSampleBm25(pos_doc_i_passages, commonWordListPos, stemming, query_length,
                                                     idf_dic, bert_num)
            elif block_match == 'bert':
                doc = calWholeDocAndSampleBert(pos_doc_i_passages, tokenModel, query_tok, query_length, bert_num,args)
            elif block_match == "random":
                doc = calWholeDocAndSampleRandomly(pos_doc_i_passages, commonWordListPos, stemming, query_length,
                                                     idf_dic, bert_num)

            doc_tok = tokenModel.tokenize(doc)
            yield qid, did, query_tok, doc_tok


def _pack_n_ship(batch, bert_num):
    # QLEN = 20
    # MAX_DLEN = 800
    QLEN = max(len(b) for b in batch['query_tok'])
    MAX_DLEN = (512 - QLEN - 3) * bert_num

    DLEN = min(MAX_DLEN, max(len(b) for b in batch['doc_tok']))
    return {
        'query_id': batch['query_id'],
        'doc_id': batch['doc_id'],
        'query_tok': _pad_crop(batch['query_tok'], QLEN),
        'doc_tok': _pad_crop(batch['doc_tok'], DLEN),
        'query_mask': _mask(batch['query_tok'], QLEN),
        'doc_mask': _mask(batch['doc_tok'], DLEN),
        'label': batch['label'],
    }


def _pad_crop(items, l):
    result = []
    for item in items:
        if len(item) < l:
            item = item + [-1] * (l - len(item))
        if len(item) > l:
            item = item[:l]
        result.append(item)
    cuda = torch.cuda.is_available()
    if cuda:
        return torch.tensor(result).long().cuda()
    else:
        return torch.tensor(result).long()


def _mask(items, l):
    result = []
    for item in items:
        # needs padding (masked)
        if len(item) < l:
            mask = [1. for _ in item] + ([0.] * (l - len(item)))
        # no padding (possible crop)
        if len(item) >= l:
            mask = [1. for _ in item[:l]]
        result.append(mask)
    cuda = torch.cuda.is_available()
    if cuda:
        return torch.tensor(result).float().cuda()
    else:
        return torch.tensor(result).float()
