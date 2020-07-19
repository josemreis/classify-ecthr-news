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
from random import choice, seed
import pycountry
import re
import pandas as pd
from googletrans import Translator
import googletrans
from expressvpn import wrapper
import subprocess
from time import sleep
from random import randint

#### Main class
class document_similarity():
    ### Sub-class: prepare the input data
    class prep_data:
        """sub-class for preping the data before the string distance calculations"""
        ### some variables
        def __init__(self):
            self.doccano_pass_path = "/home/jmr/Desktop/doccano_access.txt"
            self.doccano_baseurl = "https://label-ecthr.herokuapp.com/"
            self.doccano_username = "josemreis"
            self.langs_path = '/home/jmr/Dropbox/Current projects/thesis_papers/transparency, media, and compliance with HR Rulings/ecthr_media&compliance/data/media_data/3_classify_ecthr_news/features/data/interm_data/langs_data.csv' 
            self.articles_data_path = '/home/jmr/Dropbox/Current projects/thesis_papers/transparency, media, and compliance with HR Rulings/ecthr_media&compliance/data/media_data/3_classify_ecthr_news/annotate/data/unlabeled_news_all.csv'
            self.rulings_docs_path = '/home/jmr/Dropbox/Current projects/thesis_papers/transparency, media, and compliance with HR Rulings/ecthr_media&compliance/data/case_docs_data/rulings_data/json' 
            self.rulings_files = glob(self.rulings_docs_path + "/*JUD*")
            self.com_files = glob(self.rulings_docs_path + "/*CL*")
            self.prep_data_logfile = '/home/jmr/Dropbox/Current projects/thesis_papers/transparency, media, and compliance with HR Rulings/ecthr_media&compliance/data/media_data/3_classify_ecthr_news/features/scripts/logs/prep_input.txt'
        ### load the labeled data
        def load_labeled_articles(self):
            ## Log in to the client
            doccano_client = log_in(baseurl = self.doccano_baseurl,
                                username = self.doccano_username,
                                pswrd_path = self.doccano_pass_path)
            ## pull the labels data
            labels = labels_df(doccano_client, 1).rename(columns = {"id":"label_id",
                                                                    "text":"label_name"}).filter(items = ["label_id", "label_name"])
            ## pull the labeled docs df
            labeled_docs = get_labeled_docs(doccano_client, 1).rename(columns = {"label":"label_id"})
            ## combine
            comb = pd.merge(labels, labeled_docs, on = "label_id", how = "left")
            ## drop and rename some cols for matching with articles
            out = comb.drop(columns = ['id', 'user', 'annotation_approver'])
            out.columns = out.columns.str.replace('meta.', '')
            return out
        ### load lanugages dataset
        def load_langs_data(self):
            ## parse csv into pandas
            parsed = pd.read_csv(self.langs_path)
            parsed.columns = parsed.columns.str.replace("-", "_")
            return parsed
        ### load the articles 
        def load_corpus(self):
            ## load the dataset
            dta_all = pd.read_csv(self.articles_data_path, encoding = "utf-8").drop(columns = ['label'])
            # appno var
            dta_all['appno'] = dta_all.case_id.apply(lambda x: x.split("_")[0])
            ## standardize the languages into alpha3b, language code seemingly used by HUDOC
            # first, standardize the languages to alpha_2
            dta_all['source_lang_alpha2'] = dta_all.source_language.apply(lambda x: pycountry.languages.lookup(x).alpha_2)
            # load the language data
            langs_data = self.load_langs_data().add_prefix("source_lang_")
            # Finally, we left join them by alpha2 code
            stand_lang = pd.merge(dta_all, langs_data, how = 'inner', on = "source_lang_alpha2")
            return stand_lang
        ### load articles data
        def load_articles_data(self):
            ## load the labeled articles
            arts_labeled = self.load_labeled_articles().drop_duplicates(subset = "article_id")
            ## load all articles 
            arts_all = self.load_corpus().drop_duplicates(subset = "article_id")
            ## combine them
            merged = pd.merge(arts_all, arts_labeled, how = "left", on = ['article_id', 'case_id', 'judgment_date', 'date_published', 'text'])
            ## transform the label variable into dummy            
            merged['ecthr_label'] = np.where(merged.label_name == 'ecthr_ruling', 1, np.where(merged.label_name == 'not_ecthr_ruling', 0, None))
            # "is_labeled" var
            merged['is_labeled'] = np.where(merged_test.label_name.isnull(), 0, 1)
            return merged
        ### function for filtering decision docs
        def filter_decision_doc(self, case_id, source_lang_alpha3b, judgment = True, pref_original = True, last_resort_isocode = "FRE"):
            # case id to app number
            appno = case_id.split("_")[0].replace("/", "_")
            # lang to caps
            lang = source_lang_alpha3b.upper()
            ### DOC type
            ## look up for rulings, given language and appno
            if judgment:
                files = self.rulings_files
            else:
                print("\nGoing with the communicated notices\n")
                files = self.com_files
            ## Filtering the docs
            # first filter: by appno
            filter_appno = list(filter(lambda s: re.search(appno, s), files))
            if len(filter_appno) == 0:
                # no match with appno, stop and return a none
                out = pd.DataFrame([{'doc_path':None,'source_lang_alpha3_b':None, "case_id": case_id}])
                print(case_id.replace("_", "/") + "\nNo files in %s"%source_lang_alpha3b + " or in english!\nNo way of comparing the texts. Returning a None value\n------\n")
            else:
                # case doc is present, go on...
                # second filter: langs
                # check if original language exists
                filter_og = list(filter(lambda s: re.search(lang, s), filter_appno))
                # check if english translation exists
                filter_eng = list(filter(lambda s: re.search("ENG", s), filter_appno))
                # last resort, to be translated to english...
                filter_lr = list(filter(lambda s: re.search(last_resort_isocode, s), filter_appno))
                # Assign the doc metadata given langs and user choices
                # if available and preferred, assign local language translation
                if pref_original and len(filter_og) > 0:
                     out = pd.DataFrame([{'doc_path':filter_og[0], 'source_lang_alpha3_b': source_lang_alpha3b, "case_id": case_id}])
                # missing and available, assign english translation
                elif len(filter_eng) > 0:
                    print("No files in %s"%source_lang_alpha3b + "\nGoing to use the english version instead\n%")
                    out = pd.DataFrame([{'doc_path':filter_eng[0], 'source_lang_alpha3_b':"eng", "case_id": case_id}])
                # again, missing go for the translation "of last resort"
                elif len(filter_lr) > 0:
                    print("No files in English nor in %s"%source_lang_alpha3b + "\nGoing to use " + last_resort_isocode +  " version instead and translating it to english\n%")
                    out = pd.DataFrame([{'doc_path':filter_lr[0], 'source_lang_alpha3_b':"to_translate", "case_id": case_id}])
                # No match, look for any translation avaiable, else return None
                else:
                    try:
                        random_translation = choice(filter_appno)
                        print("No files in English nor in %s"%source_lang_alpha3b + "\nGoing to use any available version instead and translating it to english\n%")
                        out = pd.DataFrame([{'doc_path':random_translation[0], 'source_lang_alpha3_b':"to_translate", "case_id": case_id}])
                    except:
                        # no match with appno, stop and return a none
                        out = pd.DataFrame([{'doc_path':None,'source_lang_alpha3_b':None, "case_id": case_id}])
                        print(case_id.replace("_", "/") + "\nNo files in %s"%source_lang_alpha3b + " or in english!\nNo way of comparing the texts. Returning a None value\n------\n")
            return out
        ### function for loading the rulings metadata
        def load_rulings_metadata(self, debug = True):
            ## logfile
            # start the log file. Remove existing one
            logfile = self.prep_data_logfile
            if os.path.isfile(logfile):
                os.remove(logfile)
            ### For each unique case id-lang pair available in our articles_data, pull the decision doc df
            ## Get articles data
            articles_data = self.load_articles()
            ## keep unique case-lang pairs
            unique_meta = articles_data.drop_duplicates(subset = ['case_id', 'source_lang_alpha3_b'])
            ## start the loop
            #containers
            df_list = []
            missing_list = []
            for index, row in unique_meta.iterrows():
                ### pull the decision doc metadata
                decisions_df = self.filter_decision_doc(case_id = row['case_id'], source_lang_alpha3b = row['source_lang_alpha3_b'])
                if len(decisions_df[decisions_df.doc_path.notnull()]) < 1:
                    ## missing, try communications of the rulings
                    decisions_df = self.filter_decision_doc(case_id = row['case_id'], source_lang_alpha3b = row['source_lang_alpha3_b'], judgment = False)
                    if len(decisions_df[decisions_df.doc_path.notnull()]) < 1:
                        ## no doc retrieved
                        missing_list.append(row['case_id'])
                ## append to df list
                df_list.append(decisions_df)
            ## add missing cases to logfile
            with open(logfile, "w") as output:
                output.write(str("Missing decisions for cases:\n\n" + "\n".join(list(set(missing_list)))))
            # concatenate and return
            out = pd.concat(df_list)
            return out
        ### Translation funs
        ## status of vpn
        def vpn_status(self):
            p = subprocess.Popen("expressvpn status", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
            return list([str(v).replace('\\t', ' ').replace('\\n', ' ').replace('b\'', '').replace('\'', '')
                        .replace('b"', '')
                         for v in iter(p.stdout.readline, b'')])
        ## random vpn
        def random_vpn(self):
            wrapper.random_connect()
            return
        ## translate
        def translate_article(self, dataset = None):
            ## instantiate translator class
            translator = Translator(user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36")
            ## check if connected to a vpn
            if "Connected to" not in "\n".join(self.vpn_status()):
                ## connect to a random vpn
                self.random_vpn()
                sleep(randint(10, 20))
            ## concatenate the paragraphs to make the translation faster. Remove white space, paragraphs starting with a paragraph sign, and paragraphs with less than 60 characters.
            input_pars = "\n".join(list(filter(lambda x: x != None and x.startswith("©") == False and len(x) > 60, dataset.text_paragraphs.tolist())))
            ### translation
            max_attempts = 5
            attempts = 0
            print("\ntranslating...\n")
            while True:
                attempts += 1
                try:
                    ## translate
                    # source: automatic detection
                    # destination: "en"
                    # google translate only allows text up to 15000, however above 9000 we get an error. Stick with it...
                    translated = translator.translate(input_pars[:9000], dest='en')
                    sleep(randint(1, 5))
                    break
                except BaseException as e:
                    print("\ntranslation did not work, trying again\n")
                    error_message = repr(e)
                    print(error_message)
                    if attempts > max_attempts:
                        try:
                            sleep(randint(10, 20))
                            ## initiate a new instance to get another ticket
                            translator = Translator()
                            ## translate
                            translated = translator.translate(input_pars[:5000], dest='en')
                        except:
                            # rotate ip
                            self.random_vpn()
                            sleep(randint(40, 60))
                            try:
                                translated = translator.translate(input_pars[:3000], dest='en')
                            except:
                                translated = None
                                break
            ## turn translation object to text
            if isinstance(translated, googletrans.models.Translated):
                translated_text = translated.text
            else:
                translated_text = None
            return translated_text
        ### Load rulings text data
        def load_rulings(self, debug = True):
            ## load the rulings metadata
            rulings_metadata = self.load_rulings_metadata(debug = False)
            ### For each ruling doc, load the respective json dataset
            ## if not in engish or local language, translate it
            # container
            rulings_container = []
            # start the loop
            for index, row in  rulings_metadata.iterrows():
                ## get some relevant parameters
                ## load it to a pandas df
                path = row['doc_path']
                print(path)
                if "rulings_data/json" in path:
                    df_raw = pd.read_json(path, encoding = "utf-8").rename(columns = {"case_id":"appno"})
                    df_raw['case_id'] = row['case_id']
                    ## if "to_translate", translate and concatenate it
                    if row['source_lang_alpha3_b'] == "to_translate":
                        df_raw['text'] = self.translate_article(dataset = df_raw)
                        df_raw['translated'] = "1"
                        df_raw['doc_lang'] = "ENG"
                    else:
                        ## concatenate the paragraphs into "text" col. Remove white space, paragraphs starting with a paragraph sign, and paragraphs with less than 60 characters
                        df_raw['text'] = "\n".join(list(filter(lambda x: x != None and x.startswith("©") == False and len(x) > 60, df_raw.text_paragraphs)))
                        df_raw['translated'] = "0"
                    ## keep unique fils and dorp some cols
                    df_raw = df_raw.drop_duplicates(subset = ['file']).drop(columns = ['text_paragraphs'])
                    df_raw = df_raw.rename(columns = {'appno':'appno',
                                                      'doc_type':'ruling_doc_type',
                                                      'doc_lang':'ruling_doc_lang',
                                                      'file':'ruling_doc_file',
                                                      'text':'ruling_text',
                                                      'translated':'ruling_translated'})
                    ## append to container
                    rulings_container.append(df_raw)
                    if debug:
                        print(df_raw)
            ## concatenate and return
            out = pd.concat(rulings_container)
            return out
        
        
### test
if __name__  == "__main__":
    
    prep = document_similarity.prep_data()  
    ## load articles
    arts_labeled = prep.load_labeled_articles()    
    arts_all = prep.load_articles()
    # load rulings
    #rulings_loaded = prep.load_rulings()    
        
        
        
        
        
        