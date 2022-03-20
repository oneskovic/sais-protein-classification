from contextlib import redirect_stdout
from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import progressbar
from multiprocessing import Pool
from threading import Thread, Lock
import time
import numpy as np

unlabeled_data = pd.read_csv('data/test_no_labels.csv')
ids = unlabeled_data['prot_ID']
sequences = unlabeled_data['AA_sequence']
families = ['PF17482', 'PF02665', 'PF01888', 'PF04749', 'PF00639', 'PF07819', 'PF13618', 'PF01395', 'PF00890', 'PF09721', 'PF04279']

def f(index):
    post_data = {'seq':sequences[index], 'output':'xml'}
    page = requests.post('http://pfam.xfam.org/search/sequence/', data=post_data)
    job_id = re.findall('job_id="(.*?)"', page.text)[0]
    response = requests.get('http://pfam.xfam.org/search/sequence/resultset/' + job_id)
    while response.status_code != 200:
        time.sleep(0.1)
        response = requests.get('http://pfam.xfam.org/search/sequence/resultset/' + job_id)

    for family in families:
        if family in str(response.content):
            return family
    return 'None'



if __name__ == '__main__':
    bar = progressbar.ProgressBar(max_value=len(sequences), redirect_stdout=True)
    pool = Pool(8)
    correct_labels = []
    for i, family in enumerate(pool.imap(f, range(0, len(sequences)))):
        bar.update(i)
        correct_labels.append(family)
        if family == 'None':
            print(ids[i])
        print(i, family)

    unlabeled_data['prot_Pfam'] = correct_labels
    unlabeled_data.to_csv('data/test_correct_labels.csv', index=False)