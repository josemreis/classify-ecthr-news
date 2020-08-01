#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 20:49:28 2020

@author: jmr
"""
import sys
sys.path.insert(0, '/home/jmr/Dropbox/Current projects/thesis_papers/transparency, media, and compliance with HR Rulings/ecthr_media&compliance/data/media_data/3_classify_ecthr_news/annotate/scripts/helpers')
from doccano_helpers import *
from glob import glob
from random import choices, seed

### Log in to the client
doccano_client = log_in(baseurl = "https://label-ecthr.herokuapp.com/",
                        username = "josemreis",
                        pswrd_path = "/home/jmr/Desktop/doccano_access.txt")
### existing, delete all docs in the app
#delete_docs(doccano_client, 1, delete_all = True)
### upload the corpus -- BATCH 1
## upload in small chunks.
# list of paths
paths = glob('/home/jmr/Dropbox/Current projects/thesis_papers/transparency, media, and compliance with HR Rulings/ecthr_media&compliance/data/media_data/3_classify_ecthr_news/annotate/data/input_doccano/*')
# as of now, the free heroku app only allows for uploads below 10,000. Since we will note code all docs upload, lets get 11 out of the 21 docs datasets and upload those
seed(1234)
random_paths = choices(paths, k = 11)
random_paths
# upload them
upload_file(client = doccano_client,
            project_id = 1,
            file_path = random_paths,
            is_labeled = False)
### upload the corpus -- BATCH 2
## from the remaining
seed(4567)
random_paths2 = choices(paths, k = 10)
random_paths2
to_update = [value for value in random_paths2 if value not in random_paths] 
# upload them
upload_file(client = doccano_client,
            project_id = 1,
            file_path = to_update,
            is_labeled = False)

