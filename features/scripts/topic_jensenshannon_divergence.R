#### Set up
#### --------------------------------------------------------------------------------------------
### packs
require(stm)
require(tidyverse)
require(philentropy)
### dirs
parent_dir <- '/home/jmr/Dropbox/Current projects/thesis_papers/transparency, media, and compliance with HR Rulings/ecthr_media&compliance/data/media_data/3_classify_ecthr_news' 
rulart_dyad_path <- paste(parent_dir, "features", "data", "interm_data", "rulings_article_dyad_data_raw.csv.gz", sep = "/")
articles_data_path <- paste(parent_dir, "features", "data", "interm_data", "articles_data_raw.csv.gz", sep = "/")
### load the data
rulart_dyad_raw <- read_csv(rulart_dyad_path)

#### Parameter tunning for topic number
#### -------------------------------------------------------------------------------------------
t <- searchK(documents = stm::poliblog5k.docs,
         vocab = stm::poliblog5k.voc,
         K = seq(20, 200, by = 25))

p <- t$results %>%
  select(-contains("bound")) %>%
  pivot_longer(cols = c(exclus:residual), names_to = "metric_name", values_to = "metric") %>%
  ggplot(aes(K, metric, color = factor(metric_name))) +
  geom_line() + 
  facet_wrap(~metric_name, scales = "free") + 
  theme_minimal()

ggsave(filename = '/home/jmr/Dropbox/Current projects/thesis_papers/transparency, media, and compliance with HR Rulings/ecthr_media&compliance/data/media_data/3_classify_ecthr_news/features/plots/tuning_k.pdf',
       p, device = "pdf")

## Select best k based on dispersion 
best_k <- t$results %>%
  filter(residual == min(residual)) 

#### Fit the LDA
#### ------------------------------------------------------------------------------------------
### fit
lda <- stm(documents = stm::poliblog5k.docs,
           vocab = stm::poliblog5k.voc,
           K = best_k$K,
           seed = 1234,
           verbose = TRUE)

#### comparing the probability distributions
#### -------------------------------------------------------------------------------------------
### generate a document-topic probability matrix
td_gamma <- tidytext::tidy(lda, matrix = "gamma",                    
                 document_names = sample(1:300000, size = 250000)) %>%
  mutate(gamma = round(gamma, digits = 6)) %>%
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
