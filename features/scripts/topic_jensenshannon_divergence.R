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
                      ruling_doc_file)) %>%
  select(-c(article_id, contains("file"))) %>%
  distinct(id, .keep_all = TRUE)
## generate the corpus
rulart_corpus <- corpus(before_corpus,
                        docid_field = 'id',
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
### fit several lda's with different K
k_tuned <- searchK(
  documents = stm_dtm$documents,
  vocab = stm_dtm$vocab, 
  K = seq(20, 200, by = 25),
  cores = 4
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
## Select best k based on the dispersion test (taddy, 2012) 
best_k <- t$results %>%
  filter(residual == min(residual)) 

#### Fit the LDA
#### ------------------------------------------------------------------------------------------
### fit
lda <- stm(documents = stm_dtm$documents,
           vocab = stm_dtm$vocab,
           data = stm_dtm$meta,
           K = best_k$K,
           max.em.its= 75, # default
           init.type = "Spectral",
           seed = 1234,
           verbose = TRUE)
write_rds(lda, '/home/jmr/Dropbox/Current projects/thesis_papers/transparency, media, and compliance with HR Rulings/ecthr_media&compliance/data/media_data/3_classify_ecthr_news/features/models/lda_model.rds')

#### Computing the gamma distances
#### -------------------------------------------------------------------------------------------
### generate a document-topic probability matrix
td_gamma <- tidytext::tidy(lda, matrix = "gamma",                    
                 document_names = sample(1:300000, size = 250000)) %>%
  pivot_wider(
    id_cols = 1,
    names_from = topic,
    values_from = gamma,
    names_prefix = "prop_topic_"
  ) %>%
  column_to_rownames("document")

### Compute pairwise distance for each pair of topic-probability vectors (i.e. pairwise document gamma comparison)
## method based on Badene-olmodo et al, An initial Analysis of Topic-based Similarity among Scientific Documents based on their Rhetorical Discourse Parts
jsdmatrix <- JSD(as.matrix(td_gamma))
# add appropriate ids as rownames and colnames
rownames(jsdmatrix) <- rownames(td_gamma)
colnames(jsdmatrix) <- rownames(td_gamma)

#### Generate the features dataset
#### ------------------------------------------------------------------------------------------
### function extracts the jsd from two documents
pull_jsd <- function(source_doc, target_doc, jsd_matrix = jsdmatrix){
  jsd <- jsd_matrix[source_doc, target_doc]
  out <- tibble(article_id = source_doc,
                ruling_doc_file = target_doc,
                jensen_shannon_div = jsd)
  return(out)
}

## get the jsd for each article
# get the row names
rulart_jsd <- map2_df(eng_dyads$article_id, eng_dyads$ruling_doc_file, ~ pull_jsd(source_doc = .x,
                                                                                  target_doc = .y))