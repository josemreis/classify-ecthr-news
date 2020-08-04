### packs
require(text2vec)
require(dplyr)
### load the data
model_data <- readr::read_csv('/home/jmr/Dropbox/Current projects/thesis_papers/transparency, media, and compliance with HR Rulings/ecthr_media&compliance/data/media_data/3_classify_ecthr_news/model/data/model_data.csv.gz')

### train glove embeddings --------------------------------------------------------------------------
### Generate the corpus
## subset the documents
docs <- model_data$text
# Create iterator over tokens
tokens <- word_tokenizer(docs)
# Create vocabulary. Terms will be unigrams (simple words).
it <- itoken(tokens, progressbar = FALSE)
vocab <- create_vocabulary(it)
### Prune it. Min frq = 5
vocab <- prune_vocabulary(vocab, term_count_min = 5L)
### Generate term co-occurence matrix (TCM)
vectorizer <- vocab_vectorizer(vocab)
# use window of 5 for context words (-5, +5, from target)
tcm <- create_tcm(it, vectorizer, skip_grams_window = 5L)

### Glove embeddings
## can factorize the TCM via the GloVe algorithm
glove <- GlobalVectors$new(word_vectors_size = 50, vocabulary = vocab, x_max = 10)
wv <- glove$fit_transform(tcm, n_iter = 10)
## tidy
embeddings <- as_tibble(wv) %>%
  mutate(tokens = rownames(wv)) %>%
  select(tokens, everything())

### export it
readr::write_csv(embeddings, 
                 path = gzfile('/home/jmr/Dropbox/Current projects/thesis_papers/transparency, media, and compliance with HR Rulings/ecthr_media&compliance/data/media_data/3_classify_ecthr_news/model/data/interm_data/glove_tidy.csv.gz'))

readr::write_rds(wv, 
                 path = '/home/jmr/Dropbox/Current projects/thesis_papers/transparency, media, and compliance with HR Rulings/ecthr_media&compliance/data/media_data/3_classify_ecthr_news/model/data/interm_data/glove_raw.csv.gz',
                 compress = "gz")

### testing it
#Make distance (cosine) matrix
d <- dist2(wv, method="cosine")
print(dim(d))

# find closest words (lower cosine distance)
findCloseWords <- function(w,d,n) {
  words = rownames(d)
  i <- which(words==w)
  if (length(i) > 0) {
    res <- sort(d[i,])
    print(as.matrix(res[2:(n+1)]))
  } 
  else {
    print("Word not in corpus.")
  }
}

findCloseWords("law", d, 10)
findCloseWords("court", d, 10)
findCloseWords("Strasbourg", d, 10)
findCloseWords("torture", d, 10)

