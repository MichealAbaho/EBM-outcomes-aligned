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

if __name__=='__main__':
    par = argparse.ArgumentParser()
    par.add_argument('--data', default='covid.xlsx', type=str, help='source of dataset')
    par.add_argument('--pri_sec', default='primary_outcomes', help='primary outcomes or secondary outcomes')
    args = par.parse_args()
    prepare_covid_data(args)