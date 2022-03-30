from contextlib import redirect_stdout
import requests
import re
import pandas as pd
import progressbar
from multiprocessing import Pool
from threading import Thread, Lock
import time
import numpy as np

unlabeled_data = pd.read_csv('data/test-novo-no-labels.csv')
ids = unlabeled_data['prot_ID']
sequences = unlabeled_data['AA_sequence']
families = ['PF04598', 'PF02081', 'PF14124', 'PF03354', 'PF10187', 'PF11222', 'PF08885', 'PF07406', 'PF02560', 'PF03871', 'PF06239', 'PF11011', 'PF15371', 'PF07863', 'PF08708', 'PF03702', 'PF08701', 'PF16579', 'PF07564', 'PF12253', 'PF08344', 'PF16385', 'PF11449', 'PF14610', 'PF10240', 'PF14347', 'PF15884', 'PF04404', 'PF08356', 'PF12367', 'PF09270', 'PF04871', 'PF09848', 'PF04712', 'PF07034', 'PF02750', 'PF09090', 'PF17517', 'PF10486', 'PF09717', 'PF08807', 'PF06889', 'PF12842', 'PF13574', 'PF07052', 'PF11874', 'PF04721', 'PF05106', 'PF08698', 'PF04062', 'PF11006', 'PF09068', 'PF04420', 'PF08618', 'PF09538', 'PF05030', 'PF12619', 'PF14937', 'PF05219', 'PF00870', 'PF03885', 'PF15239', 'PF06395', 'PF14438', 'PF08861', 'PF04621', 'PF16173', 'PF10998', 'PF14454', 'PF06147', 'PF06050', 'PF08167', 'PF07040', 'PF14606', 'PF07297', 'PF05345', 'PF03957', 'PF10558', 'PF15785', 'PF06930', 'PF04949', 'PF05827', 'PF05499', 'PF09440', 'PF03125', 'PF06552', 'PF05527', 'PF11267', 'PF14457', 'PF11777', 'PF17261', 'PF14966', 'PF12257', 'PF15106', 'PF16103', 'PF09738', 'PF10176', 'PF05404', 'PF13934', 'PF12013', 'PF02010', 'PF05361', 'PF02505', 'PF10210', 'PF10220', 'PF15955', 'PF07231', 'PF10738', 'PF07896', 'PF11594', 'PF08293', 'PF15383', 'PF01905', 'PF07424', 'PF07165', 'PF03192', 'PF16046', 'PF04113', 'PF16507', 'PF10675', 'PF05735', 'PF15261', 'PF14390', 'PF14773', 'PF13963', 'PF01235', 'PF09392', 'PF15262', 'PF08769', 'PF07807', 'PF09968', 'PF04100', 'PF10455', 'PF04310', 'PF08854', 'PF12754', 'PF03378', 'PF06191', 'PF14007', 'PF09029', 'PF14385', 'PF10233', 'PF12070', 'PF07930', 'PF16584', 
'PF08216', 'PF00477', 'PF12957', 'PF16378', 'PF01165', 'PF12100', 'PF16516', 'PF14208', 'PF04090', 'PF10744', 'PF15249', 'PF12631', 'PF11034', 'PF02486', 'PF13798', 'PF09415', 'PF11347', 'PF11371', 'PF10077', 'PF08690', 'PF16911', 'PF06730', 'PF08832', 'PF09797', 'PF04634', 'PF12917', 'PF12527', 'PF13204', 'PF07830', 'PF16750', 'PF05659', 'PF08853', 'PF06005', 'PF12688', 'PF16266', 'PF14677', 'PF14902', 'PF12444', 'PF11201', 'PF02233', 'PF12634', 'PF14977', 'PF06124', 'PF16498', 'PF03887', 'PF14089', 'PF03759', 'PF10260', 'PF15013', 'PF17264', 'PF11677', 'PF12899', 'PF07224', 'PF10546', 'PF14354']

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