#### Set up
#### --------------------------------------------------------------------------------------------
### packs
require(stm)
require(tidyverse)
require(philentropy)
require(quanteda)
### dirs
parent_dir <- '/home/jmr/Dropbox/Current projects/thesis_papers/transparency, media, and compliance with HR Rulings/ecthr_media&compliance/data/media_data/3_classify_ecthr_news' 
rulart_dyad_path <- paste(parent_dir, "features", "data", "interm_data", "ENG_rulings_article_dyad_data_raw.csv.gz", sep = "/")
articles_data_path <- paste(parent_dir, "features", "data", "interm_data", "articles_data_raw.csv.gz", sep = "/")
rulings_data_path <- paste(parent_dir, "features", "data", "interm_data", "rulings_data_raw.csv.gz", sep = "/")
jsd_data_path <- '/home/jmr/Dropbox/Current projects/thesis_papers/transparency, media, and compliance with HR Rulings/ecthr_media&compliance/data/media_data/3_classify_ecthr_news/features/data/input/jensen-shannon-div/' 
### load the data
rulart_dyad_raw <- read_csv(rulart_dyad_path)
#### Wrangling
#### -------------------------------------------------------------------------------------------
### Preping the data for corpus building
## long-format
before_corpus <- rulart_dyad_raw %>%
  select(article_id, text, ruling_doc_file, ruling_text, appno) %>%
  pivot_longer(cols = -c(appno, contains("id"), contains("file")), names_to = "id", values_to = "text") %>%
  mutate(id = if_else(id == "text",
                      article_id, 
                      ruling_doc_file),
         id_n = row_number()) %>%
  unite("id_corpus", c(id, id_n), sep = "_", remove = FALSE) %>%
  select(-c(article_id, contains("file")))
## generate the corpus
rulart_corpus <- corpus(before_corpus,
                        docid_field = 'id_corpus',
                        text_field = 'text')
## turn to document feature matrix
dtm <- dfm(
  rulart_corpus,
  tolower = TRUE,
  stem = TRUE,
  remove = stopwords::data_stopwords_stopwordsiso$en,
  remove_punct = TRUE,
  remove_separators = TRUE,
  remove_url = TRUE,
  remove_symbols = TRUE,
  remove_numbers = TRUE,
  padding = TRUE
)
# trim
dtm_trimed <- dfm_trim(dtm, min_docfreq = .01, max_docfreq = 0.90, docfreq_type = "prop")
# convert to stm
stm_dtm <- convert(dtm_trimed, to = "stm", docvars = docvars(dtm_trimed))
# clean up..
rm(list = c("dtm", "dtm_trimed"))

#### Parameter tunning for topic number
#### -------------------------------------------------------------------------------------------
if (!file.exists('/home/jmr/Dropbox/Current projects/thesis_papers/transparency, media, and compliance with HR Rulings/ecthr_media&compliance/data/media_data/3_classify_ecthr_news/features/plots/tuning_k.pdf')) {
  ### fit several lda's with different K
  k_tuned <- searchK(
    documents = stm_dtm$documents,
    vocab = stm_dtm$vocab, 
    K = ceiling(seq(20, 80, length.out = 4)),
    verbose = TRUE
  )
  ### plot the metrics
  p <- k_tuned$results %>%
    select(-contains("bound")) %>%
    pivot_longer(cols = c(exclus:residual), names_to = "metric_name", values_to = "metric") %>%
    ggplot(aes(K, metric, color = factor(metric_name))) +
    geom_line() + 
    facet_wrap(~metric_name, scales = "free") + 
    theme_minimal()
  ggsave(filename = '/home/jmr/Dropbox/Current projects/thesis_papers/transparency, media, and compliance with HR Rulings/ecthr_media&compliance/data/media_data/3_classify_ecthr_news/features/plots/tuning_k.pdf',
         plot = p, device = "pdf")
}

#### Fit the LDA
#### ------------------------------------------------------------------------------------------
### fit
if (!file.exists('/home/jmr/Dropbox/Current projects/thesis_papers/transparency, media, and compliance with HR Rulings/ecthr_media&compliance/data/media_data/3_classify_ecthr_news/features/models/lda_model.rds')) {
  
  lda <- stm(documents = stm_dtm$documents,
             vocab = stm_dtm$vocab,
             data = stm_dtm$meta,
             K = 80,
             max.em.its= 100,
             init.type = "Spectral",
             seed = 1234,
             verbose = TRUE)
  
  write_rds(lda, '/home/jmr/Dropbox/Current projects/thesis_papers/transparency, media, and compliance with HR Rulings/ecthr_media&compliance/data/media_data/3_classify_ecthr_news/features/models/lda_model.rds')
  
} else {
  
  lda <- read_rds('/home/jmr/Dropbox/Current projects/thesis_papers/transparency, media, and compliance with HR Rulings/ecthr_media&compliance/data/media_data/3_classify_ecthr_news/features/models/lda_model.rds')
  
  
}

#### Computing the gamma distances
#### -------------------------------------------------------------------------------------------
### generate a document-topic probability matrix
td_gamma <- tidytext::tidy(lda, matrix = "gamma",                    
               document_names = names(stm_dtm$documents)) %>%
pivot_wider(
  id_cols = 1,
  names_from = topic,
  values_from = gamma,
  names_prefix = "prop_topic_"
) %>%
  rename(id_corpus = document) %>%
  mutate(id = str_remove(id_corpus, "_[0-9]+$") %>% str_trim()) %>%
  distinct(id, .keep_all = TRUE)

### Compute pairwise distance for each pair of topic-probability vectors (i.e. pairwise document gamma comparison)
## method based on Badene-olmodo et al, An initial Analysis of Topic-based Similarity among Scientific Documents based on their Rhetorical Discourse Parts
# matrix level operation made the laptop crash many times
# jsdmatrix <- JSD(as.matrix(td_gamma))
# # add appropriate ids as rownames and colnames
# rownames(jsdmatrix) <- rownames(td_gamma)
# colnames(jsdmatrix) <- rownames(td_gamma)
compute_jsd <- function(gamma_table = td_gamma, article_id, ruling_id, debug = TRUE) {
  
  ## pull the relevant topic-prob distributions
  article_probs <- gamma_table[gamma_table$id == article_id,-c(1, 82)] %>% as.matrix()
  ruling_probs <- gamma_table[gamma_table$id == ruling_id,-c(1, 82)] %>% as.matrix()
  ## compute the jensen shannon divergence
  jsd_scalar <- JSD(rbind(article_probs, ruling_probs))
  ## jsd tab
  jsd_tab <- tibble(jsd_score = jsd_scalar,
                    article_id = article_id,
                    ruling_doc_file = ruling_id)
  if (debug){
    print(jsd_tab)
  }
  return(jsd_tab)
}

## get the jsd for each article
# get the row names
rulart_jsd <- map2_df(rulart_dyad_raw$article_id, rulart_dyad_raw$ruling_doc_file, 
                      ~ compute_jsd(gamma_table = td_gamma, debug = FALSE,
                                    article_id = .x, ruling_id = .y))
# export
write_csv(rulart_jsd,
          '/home/jmr/Dropbox/Current projects/thesis_papers/transparency, media, and compliance with HR Rulings/ecthr_media&compliance/data/media_data/3_classify_ecthr_news/features/data/input/jsd_scores.csv')