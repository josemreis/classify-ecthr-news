#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 19:46:41 2020

@author: jmr
"""
import sys
sys.path.insert(0, '/home/jmr/Dropbox/Current projects/thesis_papers/transparency, media, and compliance with HR Rulings/ecthr_media&compliance/data/media_data/3_classify_ecthr_news/annotate/scripts/helpers')
from doccano_helpers import *
from glob import glob
from random import choices, seed
import pycountry
import requests
import re
#### Main class
class document_similarity():
    ### prepare the input data
    class prep_data:
        """sub-class for preping the data before the string distance calculations"""
        ### some variables
        def __init__(self):
            self.doccano_pass_path = "/home/jmr/Desktop/doccano_access.txt"
            self.doccano_baseurl = "https://label-ecthr.herokuapp.com/"
            self.doccano_username = "josemreis"
            self.langs_url = 'https://raw.githubusercontent.com/datasets/language-codes/master/data/language-codes-full.csv'
            self.articles_data_path = '/home/jmr/Dropbox/Current projects/thesis_papers/transparency, media, and compliance with HR Rulings/ecthr_media&compliance/data/media_data/3_classify_ecthr_news/annotate/data/unlabeled_news_all.csv'
            self.rulings_docs_path = '/home/jmr/Dropbox/Current projects/thesis_papers/transparency, media, and compliance with HR Rulings/ecthr_media&compliance/data/case_docs_data/rulings_data/json' 
            self.rulings_files = glob(self.rulings_docs_path + "/*JUD*")
            self.com_files = glob(self.rulings_docs_path + "/*CL*")
            self.prep_data_logfile = '/home/jmr/Dropbox/Current projects/thesis_papers/transparency, media, and compliance with HR Rulings/ecthr_media&compliance/data/media_data/3_classify_ecthr_news/features/scripts/logs/prep_input.log'
        ### load the labeled data
        def load_labels(self):
            ## Log in to the client
            doccano_client = log_in(baseurl = self.doccano_baseurl,
                                username = self.doccano_username,
                                pswrd_path = self.doccano_pass_path)
            ## pull the labels data
            labels = labels_df(doccano_client, 1)
            ## pull the labeled docs df
            labeled_docs = get_labeled_docs(doccano_client, 1)
            return labeled_docs
        ### load lanugages dataset
        def load_langs_data(self):
            ## get request
            resp = requests.get(self.langs_url)
            resp.raise_for_status()
            ## parse csv into pandas
            parsed = pd.read_csv(io.StringIO(resp.text))
            parsed.columns = parsed.columns.str.replace("-", "_")
            return parsed
        ### standardize language to alpha_2
        def stand_lang_alpha2(target_lang):
            ## identify the language codes
            all_codes = pycountry.languages.lookup(target_lang)
            return all_codes.alpha_2
        ### load the articles 
        def load_articles(self):
            ## load the dataset
            dta_all = pd.read_csv(self.articles_data_path, encoding = "utf-8").drop(columns = ['label'])
            ## standardize the languages into alpha3b, language code seemingly used by HUDOC
            # first, standardize the languages to alpha_2
            dta_all['source_lang_alpha2'] = dta_all.source_language.apply(stand_lang_alpha2)
            # load the language data
            langs_data = load_langs_data().add_prefix("source_lang_")
            # Finally, we left join them by alpha2 code
            stand_lang = pd.merge(dta_all, langs_data, how='inner', on = "source_lang_alpha2")
            return stand_lang
        ### find the the decision original language or in english
        def find_decisions(self, case_id, source_lang_alpha3b, judgment = True, pref_original = True, debug = True):
            # case id to app number
            appno = case_id.split("_")[0].replace("/", "_")
            # lang to caps
            lang = source_lang_alpha3b.upper()
            ## look up for rulings, given language and appno
            if judgment:
                files = self.rulings_files
            else:
                print("\nGoing with the communicated notices\n")
                files = self.com_files
            ## filter by appno
            filter_appno = list(filter(lambda s: re.search(appno, s), files))
            if len(filter_appno) > 0:
                ## check if original language exists
                filter_og = list(filter(lambda s: re.search(lang, s), filter_appno))
                ## check if english translation exists
                filter_eng = list(filter(lambda s: re.search("ENG", s), filter_appno))
                ## filter french
                if pref_original and len(filter_og) > 0:
                    ## stick with the original
                    out = pd.DataFrame([{'doc_path':filter_og[0], 'lang':lang, "case_id": case_id}])
                elif len(filter_eng) > 0:
                    ## return the english one
                    print("No files in %s"%lang + "\nGoing to use the english version instead\n%")
                    out = pd.DataFrame([{'doc_path':filter_og[0], 'lang':"eng", "case_id": case_id}])
                else:
                    ## return the english one
                    print("No files in %s"%lang + " or in english!\nNo way of comparing the texts. Returning a None value.\nAnd adding it to the log file at ")
                    out = pd.DataFrame([{'doc_path':None,'lang':None, "case_id": case_id}])
            else:
                out = pd.DataFrame([{'doc_path':None,'lang':None, "case_id": case_id}])
                print(appno.replace("_", "/") + "\nNo files in %s"%lang + " or in english!\nNo way of comparing the texts. Returning a None value\n------\n")
            return out
    