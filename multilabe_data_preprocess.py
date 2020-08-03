import numpy as np
import os
import sys
import re
import ast
from flair.data import Sentence
from glob import glob
import json
from argparse import ArgumentParser
import pandas as pd
#importing packages one level up
for pth in ['../../BNER-tagger/', '../']:
    sys.path.append(pth)
from datasets import taxonomy
import helper_functions as utils

core_areas, COMET_LABELS = taxonomy.comet_taxonomy()

class data_load:
    def __init__(self, dest_folder):
        self.dest_folder = dest_folder

    '''
    Extract EBM-comet multi-label annotations and convert them to the actual outcome domains from outcome taxonomy
    '''
    def read_ebm_comet(self, ebm_comet):
        data, sentence = [], []
        for i in ebm_comet:
            open_f = open(i, 'r')
            data += open_f.readlines()

        with open('{}/ebm_comet_multilabels.txt'.format(self.dest_folder), 'w') as file,\
                open('{}/ebm_comet_multilabels.json'.format(self.dest_folder), 'w') as js:

            json_output = []
            for i in data:
                if i.__contains__('docx'):
                    abstrct_title = i
                else:
                    if i != '\n':
                        if i.startswith("[['P") or i.startswith("[['E") or i.startswith("[['S") or re.search('\[\]', i):
                            multi_labels = i
                        else:
                            i = i.split()
                            sentence.append((i[0], i[1]))
                    elif i == '\n':
                        if sentence:
                            json_sent = {}
                            sent_unpacked = ' '.join(i[0] for i in sentence)
                            tag_unpacked = [i[1] for i in sentence]
                            sent = Sentence(sent_unpacked.strip())
                            outcome_domain, outcomes = [], []
                            multi_labels = ast.literal_eval(multi_labels)

                            d = k = ann = 0
                            for i in range(len(sent)):
                                if i == d:
                                    if tag_unpacked[i].startswith('B-'):
                                        z = sent[i].text
                                        # fetch taxonomy labels
                                        out_domain = multi_labels[ann]
                                        out_domain_ = [i for i in out_domain if len(i) > 2]
                                        t_tag = []
                                        for t in out_domain_:
                                            t_tag.append(fetch_taxonomy_label(annotation_label=t))

                                        file.write('{} {} {}\n'.format(sent[i].text, tag_unpacked[i], t_tag))

                                        if out_domain[0][0] not in ['E', 'S']:
                                            for j in range(i + 1, len(sent)):
                                                if tag_unpacked[j].startswith('I-'):
                                                    file.write('{} {} {}\n'.format(sent[j].text, tag_unpacked[j], t_tag))
                                                    z += ' {}'.format(sent[j].text)
                                                    d = j
                                                else:
                                                    outcomes.append(z)
                                                    break
                                        else:
                                            for j in range(i + 1, len(sent)):
                                                if re.search('E\d', tag_unpacked[j]) or re.search('S\d', tag_unpacked[j]):
                                                    z += ' ' + sent[j].text
                                                    file.write('{} {} {}\n'.format(sent[j].text, tag_unpacked[j], t_tag))
                                                    d = j
                                                    outcomes.append(z)
                                                    break
                                                elif re.search('B', tag_unpacked[j]) or ('Seperator' == tag_unpacked[j] and out_domain[0][0] == 'S'):
                                                    z = '' if out_domain[0][0] == 'S' else sent[j].text
                                                    file.write('{} {} {}\n'.format(sent[j].text, tag_unpacked[j], t_tag))
                                                else:
                                                    z += ' ' + sent[j].text
                                                    file.write('{} {} {}\n'.format(sent[j].text, tag_unpacked[j], t_tag))
                                        outcome_domain.append(t_tag)
                                        ann += 1
                                    else:
                                        file.write('{} {}\n'.format(sent[i].text, 'O'))
                                        pass
                                    d += 1
                            file.write('{}\n\n'.format(outcome_domain))
                            json_sent['sentence'] = sent_unpacked
                            json_sent['outcomes'] = outcomes
                            json_sent['multi-labels'] = outcome_domain
                            json_output.append(json_sent)

                        sentence.clear()
            json.dump(json_output, js, indent=1)

            file.close()
            js.close()

    def read_ebm_nlp(self, ebm_nlp):
        sentence = []
        r = open('{}/wrong.txt'.format(dest_folder), 'w')
        # with open(self.ebm_nlp, 'r') as open_f, \
        #         open('{}/ebm_nlp_multilabels.txt'.format(self.dest_folder), 'w') as file, \
        #         open('{}/ebm_nlp_multilabels.json'.format(self.dest_folder), 'w') as js:
        data, ground_truth = [], {}
        for i in ebm_nlp:
            if i.__contains__('.txt'):
                open_f = open(i, 'r')
                data += open_f.readlines()
            elif i.__contains__('.xlsx'):
                ground_truth = i
        #         ground_truth_csv = pd.read_csv(i)
        #         labels = set(ground_truth_csv['Label'])
        #         for label in labels:
        #             label_ = ground_truth_csv.loc[ground_truth_csv['Label'] == label.strip()]
        #             ground_truth[label.lower()] = list(label_['Outcome'])
        #
        # print(ground_truth.keys())
        # outcomes_list = [i for j in list(ground_truth.values()) for i in j]
        # for x,y in ground_truth.items():
        #     r.write('+++'+x+'+++\n')
        #     for i in y:
        #         r.write('\n{}'.format(i))
        #     r.write('\n')
        sheets = pd.ExcelFile(ground_truth)
        f = 0
        for i in data:
            if i != '\n':
                i = i.split()
                sentence.append((i[0], i[1]))
            elif i == '\n':
                sent_unpacked = [i[0] for i in sentence]
                tag_unpacked = [i[1] for i in sentence]
                d = k = 0
                #print(' '.join(sent_unpacked))
                sent_tokens, tag_tokens = [], []
                # process each word in a sentence, looking for those that form outcome phrases and obtain a vector representation for entire outcome phrase,
                for i in range(len(sent_unpacked)):
                    if i == d:
                        if tag_unpacked[i].startswith('B-'):
                            #print(tag_unpacked[i])
                            out_domain = tag_unpacked[i][2:].lower().strip()
                            z, z_indexes = sent_unpacked[i], [i]
                            sent_tokens.append(sent_unpacked[i])
                            #tag_tokens.append(tag_unpacked[i][0]+'-'+tag_unpacked[i][2:].lower())
                            for j in range(i + 1, len(sent_unpacked)):
                                if tag_unpacked[j].startswith('I-'):
                                    z += ' {}'.format(sent_unpacked[j])
                                    z_indexes.append(j)
                                    sent_tokens.append((sent_unpacked[j]))
                                    #tag_tokens.append(tag_unpacked[j][0]+'-'+tag_unpacked[j][2:].lower())
                                    d = j
                                else:
                                    break

                            primary_domain_ebm_nlp = []
                            sht_names = sheets.sheet_names

                            b = False
                            #print(z, out_domain)
                            for sht in sht_names:
                                df_multiple_domains = sheets.parse(sht)
                                df_multiple_domains = df_multiple_domains.loc[df_multiple_domains['Label'].str.lower() == sht.strip().lower()]
                                if sht.strip().lower() == out_domain:
                                    for y in df_multiple_domains.iloc[:,2:].values.tolist():
                                        y = [str(i) for i in y]
                                        label = re_label_nlp(out_domain)
                                        if y[0].strip() == z.strip():
                                            #extracted outcome is not an outcome
                                            if all(i=='nan' for i in y[1:]):
                                                print(1, y, out_domain)
                                                for m, n in enumerate(z_indexes):
                                                    tag_tokens.append('O')
                                            #the outcome is fine
                                            elif y[1] == z.strip():
                                                print(2, z, y[1], out_domain)
                                                m = 0
                                                for n in z_indexes:
                                                    if m == 0:
                                                        tag_tokens.append('B-{}'.format(label))
                                                        m += 1
                                                    else:
                                                        tag_tokens.append('I-{}'.format(label))
                                            #corrected outcome is different from original outcome
                                            elif y[1] != 'nan' and y[1] != z.strip():
                                                print(3, z, y[1], out_domain)
                                                z_ = y[1].strip().split()
                                                m = 0
                                                for n in z_indexes:
                                                    if tag_unpacked[n] == z_[m]:
                                                        if tag_unpacked[n] == z_[0]:
                                                            tag_tokens.append('B-{}'.format(label))
                                                        else:
                                                            tag_tokens.append('I-{}'.format(label))
                                                        m += 1
                                                    else:
                                                        tag_tokens.append('O')
                                            elif y[1] == 'nan':
                                                print(4, z, y[2:], out_domain)
                                                y_ = [i for i in y[2:] if i != 'nan']
                                                for u in y_:
                                                    z_ = u.strip().split()
                                                    label = core_areas[y_.index(u)]
                                                    m = 0
                                                    for n in z_indexes:
                                                        if tag_unpacked[n] == z_[m]:
                                                            if tag_unpacked[n] == z_[0]:
                                                                tag_tokens.append('B-{}'.format(label))
                                                            else:
                                                                tag_tokens.append('I-{}'.format(label))
                                                            m += 1
                                                        else:
                                                            tag_tokens.append('O')
                                            b = True
                                            break

                                    if b:
                                        break
                            print(sent_tokens)
                            print(tag_tokens)
                        else:
                            sent_tokens.append(sent_unpacked[i])
                            tag_tokens.append(tag_unpacked[i])
                            if tag_unpacked[i] != '0':
                                print('----------------{}----------------')
                                break
                        d += 1
                if len(sent_tokens) != len(tag_tokens):
                    print('\n\n\n\n\nSomething is wrong\n\n\n\n\n')
                    print(sent_tokens)
                    print(tag_tokens)
                    break
                # else:
                #     print(sent_tokens)
                #     print(tag_tokens)
                sentence.clear()
                f += 1
                if f == 30:
                    pass

def fetch_taxonomy_label(annotation_label):
    core_areas, COMET_LABELS = taxonomy.comet_taxonomy()  # taxonomy of outcomes
    domain = int(annotation_label.split()[-1])
    for k, v in COMET_LABELS.items():
        if domain in list(v.keys()):
            primary_domain_outcome = k
    return primary_domain_outcome

def jaccard_similarity(sent1, sent2):
    list1, list2 = sent1.split(), sent2.split()
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

def re_label_nlp(label,  ebm_nlp=False):
    if label in ['physical', 'pain']:
        label = core_areas[0]
    elif label == 'adverse-effects':
        label = core_areas[4]
    elif label == 'mortality':
        label = core_areas[2]
    return label



def main(args):
    data_loader = data_load(dest_folder=dest_folder)
    if args.file_name.lower() == 'ebm-nlp':
        ebm_nlp_data = [args.ebm_nlp_data + '/' + j for j in os.listdir(args.ebm_nlp_data) if not j.__contains__('labels')]
        data_loader.read_ebm_nlp(ebm_nlp=ebm_nlp_data)
    elif args.file_name.lower() == 'ebm-comet':
        ebm_comet_data = [j for j in glob('{}/*.txt'.format(args.ebm_comet_data))]
        data_loader.read_ebm_comet(ebm_comet=ebm_comet_data)

if __name__ == '__main__':
    par = ArgumentParser()
    par.add_argument('--ebm_comet_data', type=str, help='source of the ebm_comet data')
    par.add_argument('--ebm_nlp_data', type=str, help='source of the ebm_nlp data')
    par.add_argument('--dest_folder', type=str, help='destination of the pre-processed data')
    par.add_argument('--file_name', type=str, help='Is it ebm-comet or ebm-nlp')

    args = par.parse_args()
    dest_folder = utils.create_directories_per_series_des(args.dest_folder)
    main(args)

