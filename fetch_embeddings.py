# -*- coding: utf-8 -*-
# @Author: micheala
# @Created: 11/12/19
# @Contact: michealabaho265@gmail.com

import os
import argparse
import re
import numpy as np
import torch.nn as nn
import torch
import json
import pprint
import pickle
import sys
sys.path.append('../BNER-tagger/')
print(sys.path)
#import read_word_anns.py as read
from gensim.models import KeyedVectors
import ast
import pandas as pd
import data_prep as dp
from glob import glob
from numpy.linalg import norm
from scipy.spatial.distance import cosine, euclidean
from flair.embeddings import TransformerDocumentEmbeddings, TransformerWordEmbeddings, DocumentPoolEmbeddings, BertEmbeddings, FlairEmbeddings, FastTextEmbeddings, ELMoEmbeddings, StackedEmbeddings, ELMoTransformerEmbeddings, PooledFlairEmbeddings

from flair.data import Sentence
import helper_functions as utils


def fetch_embeddings(file, dest, word_map, type=None):
    weights = {}
    oov_words = 0

    if type.lower() == 'pubmed':
        store_embs = open(dest + '/{}.pickle'.format(type), 'wb')
        pubmed_vecs = KeyedVectors.load_word2vec_format(file, binary=True)
        model_vocab = list(pubmed_vecs.vocab.keys())
        word_vecs = []
        for i in model_vocab:
            d = i+' '+' '.join(str(i) for i in pubmed_vecs[i])
            word_vecs.append(d)

    elif type.lower() == 'glove':
        store_embs = open(dest + '/{}.pickle'.format(type), 'wb')
        with open(file, 'r') as f:
            word_vecs = f.readlines()
        f.close()

    elif type.lower() == 'fasttext':
        store_embs = open(dest + '/{}.pickle'.format(type), 'wb')
        with open(file, encoding='utf-8') as f:
            word_vecs = f.readlines()

    for i in word_vecs:
        line = i.split(" ")
        word = line[0]
        if word not in weights:
            try:
                vector = np.asarray(line[1:], dtype='float32')
            except Exception as e:
                print(e)
            weights[word] = vector

    weights_tensor = np.zeros((len(word_map), vector.shape[0]))

    for word,id in word_map.items():
        if word in weights:
            vec= weights.get(word)
            weights_tensor[id] = vec
        else:
            vec = np.random.rand(vector.shape[0])
            weights_tensor[id] = vec
            oov_words += 1

    word_vecs_tensors = torch.FloatTensor(weights_tensor)

    pickle.dump(word_vecs_tensors, store_embs)
    store_embs.close()

    return word_vecs_tensors

def use_flair_to_extract_context_embeddings(files, file_name, dest_folder, layer, embedding_type, embedding_size, pretrained_model=None):
    if embedding_type.lower() == 'elmo':
        context_embedding = ELMoEmbeddings(model='pubmed')
    elif embedding_type.lower() == 'elmo_transformer':
        context_embedding = ELMoTransformerEmbeddings()
    elif embedding_type.lower() == 'flair':
        context_embedding = PooledFlairEmbeddings()
    elif embedding_type.lower() == 'bioflair':
        flair_1 = PooledFlairEmbeddings('pubmed-forward')
        flair_2 = PooledFlairEmbeddings('pubmed-backward')
        elmo = ELMoEmbeddings(model='pubmed')
        context_embedding = StackedEmbeddings(embeddings=[flair_1, flair_2, elmo])
    elif embedding_type.lower() == 'biobert' or embedding_type.lower() == 'bert':
        context_embedding = TransformerWordEmbeddings(pretrained_model, layers=layer)

    data = []
    for i in files:
        open_f = open(i, 'r')
        data += open_f.readlines()
        open_f.close()

    with open('{}/{}1.pickle'.format(dest_folder, file_name), 'wb') as store, open('{}/ebm_comet_multilabels_p1.txt'.format(dest_folder), 'w') as file:
        #sentence = ''
        sentence = []
        multi_labels = ''
        instance = []
        label_representations = {}
        #fetch outcome phrase vector representations grouped in their respective outcome domain labels
        if file_name.lower() == 'ebm-comet':
            label_representations = ebm_comet_preprocessing(data=data, context_embedding=context_embedding, sentence=[], label_representations ={}, file=file)
        elif file_name.lower() == 'ebm-nlp':
            label_representations, domain_label_count = ebm_nlp_processing(data=data, context_embedding=context_embedding, sentence=[], label_representations={})

        label_centroids = {}
        print(label_representations.keys())
        print([i.shape for i in list(label_representations.values())])
        print(domain_label_count)
        #find the centroid of each group of outcome phrases vectors to represent each label
        for lab in label_representations:
            label_centroids[lab] = torch.mean(label_representations[lab], 0)
        pickle.dump(label_centroids, store)
        store.close


def ebm_comet_preprocessing(data, context_embedding, sentence, label_representations, file):
    for i in data:
        if not i.__contains__('docx'):
            if i != '\n':
                if i.startswith("[['P") or i.startswith("[['E") or i.startswith("[['S") or re.search('\[\]', i):
                    multi_labels = i
                else:
                    i = i.split()
                    sentence.append((i[0], i[1]))
            elif i == '\n':
                if sentence:
                    sent_unpacked = ' '.join(i[0] for i in sentence)
                    tag_unpacked = [i[1] for i in sentence]
                    sent = Sentence(sent_unpacked.strip())
                    context_embedding.embed(sent)
                    v = ''
                    print('\n+++++++')
                    print(sent_unpacked)
                    print(tag_unpacked)
                    print(multi_labels, type(multi_labels))
                    multi_labels = ast.literal_eval(multi_labels)

                    d = k = ann = 0
                    for i in range(len(sent)):
                        if i == d:
                            if tag_unpacked[i].startswith('B-'):
                                b = sent[i].embedding
                                b = b.reshape(1, len(b))
                                z = sent[i].text
                                file.write('{} {}\n'.format(sent[i].text, tag_unpacked[i]))
                                out_domain = multi_labels[ann]

                                if out_domain[0][0] not in ['E', 'S']:
                                    #print('hererherhehrehrehrherherherhehrehrehrherher')

                                    for j in range(i+1, len(sent)):
                                        if tag_unpacked[j].startswith('I-'):
                                            file.write('{} {}\n'.format(sent[j].text, tag_unpacked[j]))
                                            inner_b = sent[j].embedding
                                            inner_b = inner_b.reshape(1, len(inner_b))
                                            b = torch.cat((b, inner_b), dim=0)
                                            d = j
                                        else:
                                            break
                                   # print('---------',b.shape)
                                    b_mean = torch.mean(b, 0) if len(b.shape) == 2 else b
                                    b_mean = b_mean.reshape(1, len(b_mean))

                                    for dom in out_domain:
                                        if dom not in label_representations:
                                            label_representations[dom] = b_mean
                                        elif dom in label_representations:
                                            label_representations[dom] = torch.cat((label_representations[dom], b_mean), dim=0)

                                else:
                                    #print('sldncsdcnosjdlncovrsufhvksjrdhcoidfjpwoeifhekwrjbvksdjnpfcowejpfwe', )
                                    e_s_features = []
                                    x_indices = []
                                    x_indices.append((z, re_shape(sent[i].embedding)))
                                    for j in range(i + 1, len(sent)):
                                        inner_b = sent[j].embedding
                                        inner_b = inner_b.reshape(1, len(inner_b))
                                        if re.search('E\d', tag_unpacked[j]) or re.search('S\d', tag_unpacked[j]):
                                            b = torch.cat((b, inner_b), dim=0)
                                            z += ' '+sent[j].text
                                            file.write('{} {}\n'.format(sent[j].text, tag_unpacked[j]))
                                            e_s_features.append((z, tag_unpacked[j], b))
                                            x_indices.append((sent[j].text, re_shape(sent[j].embedding)))
                                            d = j
                                            break
                                        elif re.search('B', tag_unpacked[j]) or ('Seperator' == tag_unpacked[j] and out_domain[0][0] == 'S'):
                                            e_s_features.append((z, tag_unpacked[j], b))
                                            z = '' if out_domain[0][0] == 'S' else sent[j].text
                                            file.write('{} {}\n'.format(sent[j].text, tag_unpacked[j]))
                                            b = sent[j].embedding
                                            b = b.reshape(1, len(b))
                                            x_indices.append((sent[j].text, re_shape(sent[j].embedding)))
                                        else:
                                            z += ' '+sent[j].text
                                            file.write('{} {}\n'.format(sent[j].text, tag_unpacked[j]))
                                            b = torch.cat((b, inner_b), dim=0)
                                            x_indices.append((sent[j].text, re_shape(sent[j].embedding)))

                                    x = int(out_domain[0][-1])
                                    print([(i[0], i[1].shape) for i in x_indices],'+++++++++++++++++#####################+++++++++',x, [(g[0], g[1], g[2].shape) for g in e_s_features])

                                    y_indices = []
                                    if re.search('E\d', out_domain[0]):
                                        for m in range(len(e_s_features)):
                                            if m < (len(e_s_features) - 1):
                                                _m_ = e_s_features[m][2]
                                                for t in range(x):
                                                    #print('Ennnnnnnnnnnnd',e_s_features[m][0])
                                                    _m_ = torch.cat((_m_, x_indices[-(t+1)][1]), dim=0)
                                                y_indices.append(_m_)
                                        y_indices.append(e_s_features[-1][2])
                                    elif re.search('S\d', out_domain[0]):
                                        for m in range(len(e_s_features)):
                                            if m > 0:
                                                _m_ = e_s_features[m][2]
                                                for t in range(x):
                                                    _m_ = torch.cat((x_indices[t][1], _m_))
                                                    #print('Staaaaaaaaaaaaaart',e_s_features[m][0])
                                                y_indices.append(_m_)
                                        y_indices.insert(0, e_s_features[0][2])

                                    # print('---------',b.shape)
                                    b_mean = []
                                    #print(e_s_features[-1][0])

                                    for d_ in y_indices:
                                        #print('d_ shape',d_.shape)
                                        d_ = torch.mean(d_, 0) if len(d_.shape) > 1 else d_
                                        b_mean.append(d_.reshape(1, len(d_)))

                                    for b_,dom in zip(b_mean, out_domain[1:]):
                                        if dom not in label_representations:
                                            #print('final', b_.shape)
                                            label_representations[dom] = b_
                                        elif dom in label_representations:
                                            #print('final already in', b_.shape)
                                            label_representations[dom] = torch.cat((label_representations[dom], b_), dim=0)
                                ann += 1

                            else:
                                file.write('{} {}\n'.format(sent[i].text,'O'))
                                pass
                            d += 1
                    file.write('\n')
                sentence.clear()
    return label_representations

def ebm_nlp_processing(data, context_embedding, sentence, label_representations):
    domain_label_count = {}
    pain_mortality_domain = 'PAIN_MORT'
    for i in data:
        if i != '\n':
            i = i.split()
            #sentence += ' '+i[0]
            sentence.append((i[0], i[1]))
        elif i == '\n':
            # join words making up a sentence, the list of tags in each sentence and obtain the context vectors for the sentence words
            sent_unpacked = ' '.join(i[0] for i in sentence)
            tag_unpacked = [i[1] for i in sentence]
            sent = Sentence(sent_unpacked.strip())
            context_embedding.embed(sent)
            v = ''
            print('+++++++',len(sent), len(sentence))
            print(sent_unpacked)
            print(tag_unpacked)
            d = k = 0
            #process each word in a sentence, looking for those that form outcome phrases and obtain a vector representation for entire outcome phrase,
            for i in range(len(sent)):
                #print(i, d, sent[i].text)
                if i == d:
                    if tag_unpacked[i].startswith('B-'):
                        b = sent[i].embedding
                        b = b.reshape(1, len(b))
                        out_domain = tag_unpacked[i][2:].strip()

                        for j in range(i+1, len(sent)):
                            if tag_unpacked[j].startswith('I-'):
                                inner_b = sent[j].embedding
                                inner_b = inner_b.reshape(1, len(inner_b))
                                b = torch.cat((b, inner_b), dim=0)
                                d = j
                            else:
                                break
                        b_mean = torch.mean(b, 0) if len(b.shape) == 2 else b #extract the centroid for word vectors of an outcome phrase
                        b_mean = b_mean.reshape(1, len(b_mean))
                        if out_domain not in label_representations:
                            label_representations[out_domain] = b_mean
                            domain_label_count[out_domain] = 1
                        elif out_domain in label_representations:
                            label_representations[out_domain] = torch.cat((label_representations[out_domain], b_mean), dim=0)
                            domain_label_count[out_domain] += 1

                        #combine pain and mortality outcomes
                        if out_domain.lower() in ['pain', 'mortality']:
                            if pain_mortality_domain not in label_representations:
                                label_representations[pain_mortality_domain] = b_mean
                            else:
                                label_representations[pain_mortality_domain] = torch.cat((label_representations[out_domain], b_mean), dim=0)
                    else:
                        pass
                    d += 1
            sentence.clear()
    return label_representations, domain_label_count


def re_shape(x):
    l = len(x)
    return x.reshape(1, l)

def add_up_missing(x, v, direction='E'):
    v = [i[1] for i in v]
    if direction == 'S':
        v = v[:-x]
    else:
        v = v[-x:]
    d = 0
    for u in range(len(v)):
        d = torch.cat((d, v[u]), dim=0)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, help="The input data dir. Should contain the training files.")
    parser.add_argument("--file_name", default='ebm_nlp', type=str, required=True, help="name of output file.")
    parser.add_argument("--pretrained_model", default='bert-base-multilingual-cased')
    parser.add_argument("--mode", default='train', type=str, help="feature dimension")
    parser.add_argument("--pooling", default='mean', type=str, help="pooling operation")
    parser.add_argument("--norm", default='l2', type=str, help="vector norm operation")
    parser.add_argument("--layer", default='all', type=str, help="model layer")
    parser.add_argument("--embedding_type", default=None, required=True, type=str, help='which embeddings to create')
    parser.add_argument("--embedding_size", default=768, type=int, help='size of the embedding')
    parser.add_argument("--dest_folder", default=None, required=True, help='where to store the embeddings created')

    args = parser.parse_args()
    dest_folder = utils.create_directories_per_series_des(args.dest_folder)
    with open(os.path.join(dest_folder, 'config.txt'), 'w') as cf:
        for k, v in vars(args).items():
            cf.write('{}: {}\n'.format(k, v))
        cf.close()

    if args.embedding_type.lower() in ['pubmed', "glove", "fasttext"]:
        data_files = [j for j in glob('{}/*.bmes'.format(args.data_dir))]
        word_count, index2word, num_words, num_pos, line_pairs, word_map, tag_map, pos_map, index2tag, size = dp.prepare_tensor_pairs(
                                                                                                                file_path=data_files,
                                                                                                                file_out=dest_folder,
                                                                                                                mode=args.mode)
        weights = fetch_embeddings(file=args.file_name, dest=args.dest_folder, word_map=word_map, type=args.embedding_type)
    elif args.embedding_type.lower() in ["bert", "bioflair", "elmo", "biobert"]:
        if args.file_name.lower() == 'ebm-nlp':
            data_files = [j for j in glob('{}/*.txt'.format(args.data_dir)) if not j.__contains__('labels')]
        elif args.file_name.lower() == 'ebm-comet':
            data_files = [j for j in glob('{}/*.txt'.format(args.data_dir)) if not j.__contains__('comet-dataset')]
            print(data_files)
        use_flair_to_extract_context_embeddings(files=data_files,
                                                file_name=args.file_name,
                                                dest_folder=dest_folder,
                                                embedding_type=args.embedding_type,
                                                layer=args.layer,
                                                embedding_size=args.embedding_size,
                                                pretrained_model=args.pretrained_model)
    