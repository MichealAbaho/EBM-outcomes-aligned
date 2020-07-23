import numpy as np
import pandas as pd
import os
import argparse
import sys
sys.path.append(os.path.abspath('../../BNER-tagger/'))
print(sys.path)
import data_prep as dp

sent_splitter_model = dp.spacyModel()

def prepare_covid_data(args):
    covid_df = pd.read_excel(args.data)
    count = 1
    with open('{}.txt'.format(args.pri_sec), 'w') as cvd, open('{}_count.txt'.format(args.pri_sec), 'w') as ct:
        outcome = 'out_primary_measure' if args.pri_sec == 'primary_outcomes' else 'out_secondary_measure'
        covid_df[outcome] = covid_df[outcome].astype(str)
        for sent in covid_df[outcome]:
            ct.write(str(count)+'\n')
            sent_split = sent_splitter_model(sent.replace(';', ' '))
            for i,sub_sent in enumerate(sent_split.sents):
                sub_sent = sub_sent.text.split()
                for s in sub_sent:
                    cvd.write('{}\n'.format(s))
            cvd.write('\n')
            count += 1
        cvd.close()

def prepare_covid_quot_seperated_data(args):
    covid_df = pd.read_excel(args.data)
    count = 1
    covid_df_processed = []
    outcome = 'out_primary_measure' if args.pri_sec == 'primary_outcomes' else 'out_secondary_measure'
    covid_df[outcome] = covid_df[outcome].astype(str)
    study_id = covid_df.columns[0]
    for sent in covid_df[[study_id,outcome]].values:
        sent_split = sent[1].split(';')
        sent_split = [i for i in sent_split if i]
        for i,sub_sent in enumerate(sent_split):
            if sub_sent != '\n' or sub_sent != '':
                covid_df_processed.append((str(sent[0]).strip(), sub_sent.strip()))
        count += 1
    covid_df_processed_frame = pd.DataFrame(covid_df_processed, columns=['study_id', 'outcome'])
    covid_df_processed_frame.to_csv('{}_quot.csv'.format(args.pri_sec))



if __name__=='__main__':
    par = argparse.ArgumentParser()
    par.add_argument('--data', default='covid.xlsx', type=str, help='source of dataset')
    par.add_argument('--quot_separated_outcomes', action='store_true', help='dataset cotains outcomes however they are separated by quotation marks')
    par.add_argument('--pri_sec', default='primary_outcomes', help='primary outcomes or secondary outcomes')
    args = par.parse_args()
    if not args.quot_separated_outcomes:
        prepare_covid_data2(args)
    else:
        prepare_covid_quot_seperated_data(args)
