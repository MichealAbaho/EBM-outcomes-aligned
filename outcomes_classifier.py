# -*- coding: utf-8 -*-
# @Author: micheala
# @Created: 23/06/20
# @Contact: michealabaho265@gmail.com

from torch.optim.adam import Adam
from torch.optim.sgd import SGD
import csv
from flair.data import Corpus, Sentence
from flair.datasets import TREC_6, CONLL_2000, UD_ENGLISH, CSVClassificationCorpus, ClassificationCorpus, ColumnCorpus
from flair.embeddings import TransformerDocumentEmbeddings, TransformerWordEmbeddings, WordEmbeddings, StackedEmbeddings
from flair.models import TextClassifier, SequenceTagger
from flair.trainers import ModelTrainer
from flair.trainers.language_model_trainer import TextCorpus
from glob import glob
import argparse
from datasets import taxonomy
import pandas as pd
import numpy as np
from tabulate import tabulate
import math
import re
import os

def outcome_data_merger():
    core_areas, COMET_LABELS = taxonomy.comet_taxonomy()  # taxonomy of outcomes
    def fetch_taxonomy_label(annotation_label):
        domain = int(annotation_label.split()[-1])
        for k, v in COMET_LABELS.items():
            if domain in list(v.keys()):
                primary_domain_outcome = k
        return primary_domain_outcome

    csv_or_xls_files = glob(dataset_folder+'/*.xlsx')
    for i in csv_or_xls_files:
        if i.lower().__contains__('comet'):
            df_multiple_domain = pd.read_excel(i) #dataset has an outcome iwht multiple domain labelling
            primary_domain_comet = []
            columns_names = df_multiple_domain.columns.tolist()
            df_multiple_domain.fillna('X', inplace=True) #replace unfilled values with an identifier 'X'
            for row in df_multiple_domain.values.tolist():
                #clean up the entries before sorting them appropriately
                row = [i.strip() for i in row]
                row[1] = row[1].strip("['']").strip()
                row[2] = re.sub('\s+', ' ', row[2])
                #print(row)
                if row[-2] == 'X' and row[-1] == 'X': #if no primary domain exists and no comment on annotation
                    _row_ = row[:-2]
                elif row[-2] != 'X' and row[-1] == 'X': #if a primary domain exists, insert it but keep secondary domains
                    row[-2] = row[-2].strip("['']").strip()
                    row_ = [i.strip().strip("'") for i in row[1].split(',')]
                    if row[-2] in row_:
                        row_.remove(row[-2])
                    row_.insert(0, row[-2])
                    row.remove(row[1])
                    for u in range(len(row_)-1, -1, -1):
                        row.insert(1, row_[u])
                    _row_ = row[:-2]
                else:
                    _row_ = row[:-2]
                primar_domain_label = fetch_taxonomy_label(annotation_label=_row_[1])
                _row_.insert(1, primar_domain_label)
                #fix some labels with double and single quots all on one string
                if len(_row_[2]) > 4:
                    dt = [i.replace("'", '').strip() for i in _row_[-2].split(',')]
                    _row_.remove(_row_[-2])
                    _row_ = _row_ + dt
                    _row_ [-1], _row_ [-2], _row_ [-3] = _row_ [-3], _row_ [-1], _row_ [-2]
                #insert dummy nan values as labels inorder to have the minimum amount of labels for each outcome
                if len(_row_) < 6:
                    for i in range(6 - len(_row_)):
                        _row_.insert(len(_row_)-1, np.NAN)
                primary_domain_comet.append(tuple(_row_))
        else:
            sheets = pd.ExcelFile(i)
            primary_domain_ebm_nlp = []
            sht_names = sheets.sheet_names
            for sht in sht_names:
                df_multiple_domains = sheets.parse(sht)
                df_multiple_domains = df_multiple_domains.loc[df_multiple_domains['Label'].str.lower() == sht.strip().lower()]
                if sht.strip().lower() != 'ebm_nlp_outcomes':
                    #deal with each corrected outcome domain independently
                    for y in df_multiple_domains.iloc[:,2:].values.tolist():
                        if sht.strip().lower() == 'mortality':
                            y[0], y[1], y[2], y[3] = y[1], y[3], y[0], y[2]
                        if sht.strip().lower() == 'pain':
                            if str(y[0]) != 'nan' and str(y[1]) == 'nan':
                                y[0] = y[0]
                            elif str(y[0]) == 'nan' and str(y[1]) != 'nan':
                                y[0] = y[1]
                            y.remove(y[1])
                            y[2], y[3], y[4] = y[4], y[2], y[3]
                        if sht.strip().lower() == 'adverse-effects':
                            y[0], y[1], y[3], y[4] = y[1], y[3], y[4], y[0]
                        y_copy = y.copy()
                        several = False
                        if len([i for i in y if str(i) != 'nan']) > 1:
                            several = True
                            #print(y, sht)
                        if any(str(i)!='nan' for i in y_copy):
                            for o in range(len(y_copy)):
                                if str(y_copy[o]) != 'nan':
                                    #split multiple outcomes via the semi colon
                                    for m in y_copy[o].split(';'):
                                        y_len = len(y)
                                        y.clear()
                                        y = [np.NAN] * y_len
                                        y[o] = m.strip()
                                        if several:
                                            pass
                                            #print(y)
                                        y.insert(0, core_areas[o])  # insert the primary domain
                                        _y_ = [i for i in y if str(i) != 'nan']
                                        primary_domain_ebm_nlp.append(tuple(_y_))
                                        y = y[1:]

    primary_domain_comet_frame = pd.DataFrame(primary_domain_comet, columns=['Abstract', 'Label', 'Label_1', 'Label_2', 'Label_3', 'Outcome'])
    print(max([len(i) for i in primary_domain_ebm_nlp]), '\n', primary_domain_ebm_nlp[4])
    #print(tabulate(primary_domain_comet_frame.head(), headers='keys', tablefmt='psql'))
    primary_domain_ebm_nlp_frame = pd.DataFrame(primary_domain_ebm_nlp, columns=['Label', 'Outcome'])
    #print(tabulate(primary_domain_ebm_nlp_frame.head(30), headers='keys', tablefmt='psql'))
    primary_domain_comet_frame.to_csv(dest_folder+'/ebm-comet-outcomes.csv')
    primary_domain_ebm_nlp_frame.to_csv(dest_folder+'/ebm-nlp-outcomes.csv')
    #merge the dataframes holding outcmes from different datasets
    merged_outcomes = pd.concat([primary_domain_comet_frame, primary_domain_ebm_nlp_frame], ignore_index=True, sort=False)
    print(len(primary_domain_comet_frame), len(primary_domain_ebm_nlp_frame))
    return merged_outcomes

def create_dataset(data, tsv=False):
    if type(data) == str:
        data = pd.read_csv(data) if data.__contain__('.csv') else pd.read_excel(data)
    data = data[['Label', 'Outcome']].rename(columns={'Label':'label', 'Outcome':'text'})
    #prepare tsv files
    def tsv_files(df):
        df[1]['label'] = '__label__' + df[1]['label'].str.upper()
        with open(dest_folder + '/{}.txt'.format(df[0]), 'w') as t:
            for i, j in zip(df[1]['label'], df[1]['text']):
                t.write('{}\t{}'.format(i, j) + '\n')
            t.close()

    #shuffle data and remove duplicates
    data_listed = data.values.tolist()
    #print(len(df_listed))
    data_no_duplicates = []
    for i in data_listed:
        if i not in data_no_duplicates:
            data_no_duplicates.append(i)

    indices = np.random.permutation(len(data_no_duplicates))
    data_shuffled = [data_no_duplicates[i] for i in indices]
    data_ = pd.DataFrame(data_shuffled, columns=['label', 'text'])
    data_.to_csv(dest_folder+'/ebm-nlp-comet-combined-shuffled.csv')

    if not tsv:
        data_.iloc[0:int(len(data_) * 0.8)].to_csv(dest_folder + '/train.csv', sep='\t', index=False)
        data_.iloc[int(len(data_) * 0.8):int(len(data_) * 0.9)].to_csv(dest_folder + '/test.csv', sep='\t', index=False, header=False)
        data_.iloc[int(len(data_) * 0.9):].to_csv(dest_folder + '/dev.csv', sep='\t', index=False, header=False)
    else:
        train = data_.iloc[0:int(len(data_) * 0.8)]
        test = data_.iloc[int(len(data_) * 0.8):int(len(data_) * 0.9)]
        dev = data_.iloc[int(len(data_) * 0.9):]
        print('train size: {} dev size: {} and test size: {}'.format(len(train), len(dev), len(test)))
        for i in [('train', train), ('test', test), ('dev', dev)]:
            tsv_files(i)

def train_classifier(pre_trained_model, layer, lr, batch_size, pooling_sub_token, epochs, hidden_size, word_level=False, task='text_classification'):
    # corpus = NLPTaskDataFetcher.load_classification_corpus(data_folder='label_embs_2/', test_file='test.csv', train_file='train.csv', dev_file='dev.csv')
    if not word_level:
        document_embeddings = TransformerDocumentEmbeddings(pre_trained_model, fine_tune=True)
    else:
        token_embeddings = TransformerWordEmbeddings(pre_trained_model, layers=layer, pooling_operation=pooling_sub_token, fine_tune=True)

    #text classification
    if task == 'text_classification':
        corpus: Corpus = ClassificationCorpus(data_folder=dataset_folder, test_file='test.txt', dev_file='dev.txt', train_file='train.txt')
        label_dict = corpus.make_label_dictionary()
        classifier = TextClassifier(document_embeddings=token_embeddings, label_dictionary=label_dict, multi_label=False)
        # trainer = ModelTrainer(model=classifier, corpus=corpus, optimizer=SGD)
    #sequence labelling
    elif task == 'sequence_labelling':
        columns = {0: 'text', 1: 'tag'}
        corpus: Corpus = ColumnCorpus(dataset_folder, columns, train_file='train.txt', test_file='test.txt', dev_file='dev.txt')
        token_tag_dictionary = corpus.make_tag_dictionary(tag_type=columns[1])
        embedding_types = [
            TransformerWordEmbeddings(pre_trained_model, layers=layer, pooling_operation=pooling_sub_token, fine_tune=True)
        ]
        embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)
        classifier: SequenceTagger = SequenceTagger(hidden_size=hidden_size,
                                                   embeddings=embeddings,
                                                   tag_dictionary=token_tag_dictionary,
                                                   tag_type=columns[1],
                                                   use_crf=True)
    trainer: ModelTrainer = ModelTrainer(model=classifier, corpus=corpus, optimizer=SGD)
    trainer.train(dest_folder+'/{}-output'.format(task),
                  learning_rate=lr,
                  mini_batch_size=batch_size,
                  max_epochs=epochs)

def predict_labels(data, pretrained_model):
    classifier = TextClassifier.load(pretrained_model)
    data_to_classify, sub_outcomes = [], []

    with open(data, 'r') as d:
        instances = []
        for i in d.readlines():
            if i != '\n':
                i = i.split()
                instances.append(i)
            else:
                # if instances:
                instances_copy = instances.copy()
                data_to_classify.append(instances_copy)
                outcome, sub_instances, l = '', [], 0
                _outcomes_ = ()
                #print(instances_copy)
                for x in range(len(instances_copy)):
                    if x == l:
                        x_str = instances_copy[x][1]
                        #     if x_str != 'O':
                        if x_str.startswith('B') or x_str.startswith('I'):
                            outcome = instances_copy[x][0]
                            if x == len(instances_copy) - 1:
                                if str(outcome.strip()) != 'nan':
                                    sent = Sentence(outcome)
                                    classifier.predict(sent)
                                    sub_instances.append('{}:{}'.format(outcome.strip(), sent.labels[0].value))
                            else:
                                for y in range(x + 1, len(instances_copy)):
                                    if not instances_copy[y][1].startswith('B') and instances_copy[y][1] != 'O':
                                        outcome += ' {}'.format(instances_copy[y][0])
                                        outcome = outcome.strip()
                                        if y == len(instances_copy) - 1:
                                            outcome_copy = outcome
                                            if str(outcome_copy.strip()) != 'nan':
                                                sent = Sentence(outcome_copy)
                                                classifier.predict(sent)
                                                sub_instances.append('{}:{}'.format(outcome_copy.strip(), sent.labels[0].value))
                                            outcome = ''
                                        l = y
                                    else:
                                        if outcome:
                                            outcome_copy = outcome
                                            if str(outcome_copy.strip()) != 'nan':
                                                sent = Sentence(outcome_copy)
                                                classifier.predict(sent)
                                                sub_instances.append('{}:{}'.format(outcome_copy.strip(), sent.labels[0].value))
                                            outcome = ''
                                        break
                        l += 1
                sub_outcomes.append(tuple(sub_instances))
                instances.clear()
    data_to_classify = [' '.join(j[0] for j in i) for i in data_to_classify]
    data_to_classify_frame = pd.DataFrame(data_to_classify, columns=['Abstract'])
    max_outcomes_per_sentence = max([len(i) for i in sub_outcomes])
    columns_ = ['Outcome {}'.format(i+1) for i in range(max_outcomes_per_sentence)]
    sub_outcomes_frame = pd.DataFrame(sub_outcomes, columns=columns_)
    data_to_classify_frame = pd.concat([data_to_classify_frame, sub_outcomes_frame], axis=1)
    print(tabulate(data_to_classify_frame, headers='keys', tablefmt='psql'))
    #data_to_classify_frame.to_csv(os.path.dirname(data)+'/{}.csv'.format(os.path.basename(data).split('.')[0]))
    #print(tabulate(data_to_classify_frame.head(20), headers='keys', tablefmt='psql'))

def predict_labels_quot_serperated_data(data, pretrained_model):
    classifier = TextClassifier.load(pretrained_model)
    quot_separated_data = []

    d = pd.read_csv(data)
    for sent in d[['study_id','outcome']].values:
        if sent[1] != '\n' and str(sent[1]).strip() != 'nan' and type(sent[1]) != float:
            print(sent[1])
            sent_ = Sentence(sent[1].strip())
            classifier.predict(sent_)
            lab = sent_.labels
            quot_separated_data.append((sent[0], sent[1], lab))

    quot_separated_data_frame = pd.DataFrame(quot_separated_data, columns=['study_id', 'outcome', 'outcome_type'])
    quot_separated_data_frame.to_csv(os.path.join(os.path.dirname(data), '{}_predictions.csv'.format(os.path.basename(data).split('.')[0])))

if __name__ == '__main__':
    par = argparse.ArgumentParser()
    par.add_argument('--pretrained_model', default='bert-base-uncased', help='source of pretrained model')
    par.add_argument('--word_level', action='store_true', help='True: encode input sentence by aggregating embeddings for each word otherwise: encode whole sentence')
    par.add_argument('--layer', default='all',  help='layers to extract features')
    par.add_argument('--learning_rate', default=0.1, type=float, help='learning rate')
    par.add_argument('--batch_size', default=16, type=int, help='batch size')
    par.add_argument('--hidden_size', default=256, type=int, help='hidden size')
    par.add_argument('--pooling', default='mean',  help='pooling operation to perform over sub tokens')
    par.add_argument('--epochs', default=1, type=int, help='number of training epochs')
    par.add_argument('--datasets', default='datasets', help='source of data')
    par.add_argument('--dest_folder', default='classification files folder', help='source of files or data to be used for classification training')
    par.add_argument('--tsv', action='store_true', help='tab seperated text files as classification sets otherwise csv')
    par.add_argument('--quot_separated_outcomes', action='store_true', help='dataset cotains outcomes however they are separated by quotation marks')
    par.add_argument('--predictor', action='store_true', help='predictor')
    par.add_argument('--sequence_labelling', action='store_true', help='Training a sequence labelling model')

    args = par.parse_args()

    if not args.predictor:
        dataset_folder = args.datasets
        dest_folder = os.path.abspath(os.path.join(dataset_folder, args.dest_folder))
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
        if args.sequence_labelling:
            #train a sequence labelling model
            train_classifier(pre_trained_model=args.pretrained_model,
                             layer=args.layer,
                             lr=args.learning_rate,
                             batch_size=args.batch_size,
                             pooling_sub_token=args.pooling,
                             epochs=args.epochs,
                             hidden_size=args.hidden_size,
                             word_level=args.word_level,
                             task='sequence_labelling')
        else:
            #train a text classification model
            merged_outcomes = outcome_data_merger()
            merged_outcomes.to_csv(dest_folder+'/ebm-nlp-comet-combined.csv')
            print(tabulate(merged_outcomes.head(), headers='keys', tablefmt='psql'))
            create_dataset(data=merged_outcomes, tsv=args.tsv)
            print('\n\n*******************\nfinished\n*******************\n\n')
            train_classifier(pre_trained_model=args.pretrained_model, layer=args.layer, pooling_sub_token=args.pooling, epochs=args.epochs, word_level=args.word_level)
    else:
        if not args.quot_separated_outcomes:
            predict_labels(data=args.datasets, pretrained_model=args.pretrained_model)
        else:
            predict_labels_quot_serperated_data(data=args.datasets, pretrained_model=args.pretrained_model)
