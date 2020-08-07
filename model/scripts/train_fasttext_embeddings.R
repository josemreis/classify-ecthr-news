### packs
library(fastrtext)
require(tidyverse)
require(tm)
### load the data
unlab_data <- readr::read_csv('/home/jmr/Dropbox/Current projects/thesis_papers/transparency, media, and compliance with HR Rulings/ecthr_media&compliance/data/media_data/3_classify_ecthr_news/annotate/data/unlabeled_news_all.csv') %>%
  select(text)
lab_data <- readr::read_csv('/home/jmr/Dropbox/Current projects/thesis_papers/transparency, media, and compliance with HR Rulings/ecthr_media&compliance/data/media_data/3_classify_ecthr_news/model/data/model_data.csv.gz') %>%
  select(text)
corpus_data <- bind_rows(unlab_data, lab_data) %>%
  distinct(text)

### Pre-processing
corpus <- Corpus(VectorSource(corpus_data$text))
summary(corpus)
# remove punctuation
corpus <- tm_map(corpus, content_transformer(removePunctuation))
# to lower
corpus <- tm_map(corpus, content_transformer(tolower))
# remove numbers
corpus <- tm_map(corpus, content_transformer(removeNumbers))
# remove control 
text <- as.character(corpus)
## set working directory at model path
setwd('/home/jmr/Dropbox/Current projects/thesis_papers/transparency, media, and compliance with HR Rulings/ecthr_media&compliance/data/media_data/3_classify_ecthr_news/model/data/interm_data/')

### train the model
## word vectors of size 100
## skipgram model
## word ngram = 1
## min length of char ngram 2
## context window size 5 (-5; +5)
### Build the vectors
model_file <- build_vectors(text, 
                            'fasttext_emb',
                            dim = 100,
                            modeltype = "skipgram",
                            minn = 2,
                            ws = 5)
## Load the model
model <- load_model(model_file)

### get the word vectors
wv <- get_word_vectors(model)
## tidy
embeddings <- as_tibble(wv) %>%
  mutate(tokens = rownames(wv)) %>%
  select(tokens, everything())

### Export
readr::write_csv(embeddings, 
                 path = gzfile('/home/jmr/Dropbox/Current projects/thesis_papers/transparency, media, and compliance with HR Rulings/ecthr_media&compliance/data/media_data/3_classify_ecthr_news/model/data/interm_data/fasttext_tidy.csv.gz'))
readr::write_rds(wv, 
                 path = '/home/jmr/Dropbox/Current projects/thesis_papers/transparency, media, and compliance with HR Rulings/ecthr_media&compliance/data/media_data/3_classify_ecthr_news/model/data/interm_data/fasttext_raw.rds.gz',
                 compress = "gz")

### test quality
get_nn(model, word = "human", k = 100)
get_nn(model, word = "prison", k = 100)
