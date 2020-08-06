#### Set up -------------------------------------------------------------------------
### Packs
require(mlr3)
require(mlr3learners)
require(tidymodels)
require(textrecipes)
require(tidyverse)

### wd at path location
dirname(rstudioapi::getActiveDocumentContext()$path)

### Data
## model data
model_data <- read_csv("../../data/model_data.csv.gz") %>%
  mutate(model_id = paste(case_id, article_id, sep = "_"),
         ecthr_label = as_factor(ecthr_label)) %>%
  select(ecthr_label, text, article_nchar, date_distance, cosine_similarity_tf, jaccard_distance, lang_og = source_lang_alpha2, contains("ner"), country_match_ratio, model_id) %>%
  filter(article_nchar > 500) %>%
  distinct(text, .keep_all = TRUE) %>%
  drop_na()

### Pre-process
## tfidf dataset (check Rmd)
tfidf_main <- recipe(
  ecthr_label ~ ., 
  data = model_data
  ) %>%
  update_role(model_id, new_role = "id variable") %>%
  step_rm(c(lang_og, model_id)) %>% ## remove the language variable (see above)
  step_mutate( ## discretize the date distance variable
    date_distance = date_distance %>% 
      str_replace("-", "minus_") %>% 
      as_factor()
  ) %>%
  step_dummy(date_distance) %>%
  step_log(article_nchar) %>% # natural log of artice nchar
  step_tokenize(text) %>% ## remove stopwords
  step_stopwords(text, keep = FALSE, stopword_source = "stopwords-iso") %>%
  step_stem(text) %>% ## turn to ngram
  step_untokenize(text) %>% 
  step_tokenize(text, token = "ngrams", options = list(n = 3, n_min = 1, ngram_delim = "_")) %>% 
  step_tokenfilter(text, min_times = 2, max_tokens = 1000) %>%
  step_normalize(all_numeric(), -article_nchar, -all_outcomes()) %>%
  step_zv(all_predictors()) %>% # remove 0 variance vars %>%
  step_tfidf(text) %>%
  prep(model_data) ## the recipe

tfidf_main
## glove embeddings dataset 
# load the glove embeddings trained in our corpus
emb_trained <- read_csv('/home/jmr/Dropbox/Current projects/thesis_papers/transparency, media, and compliance with HR Rulings/ecthr_media&compliance/data/media_data/3_classify_ecthr_news/model/data/interm_data/glove_tidy.csv.gz')
## glove_recipe
glove_main <- recipe(
  ecthr_label ~ .,
  data = model_data
  ) %>%
  update_role(model_id, new_role = "id variable") %>%
  step_rm(c(lang_og, model_id)) %>% ## remove the language variable (see above)
  step_mutate(## discretize the date distance variable
    date_distance = date_distance %>% str_replace("-", "minus_") %>% as_factor()
  ) %>%
  step_dummy(date_distance) %>%
  step_log(article_nchar) %>%
  step_tokenize(text) %>%
  step_word_embeddings(text, embeddings = emb_trained) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric(), -article_nchar, -all_outcomes()) %>%
  prep(model_data)

glove_main
### Bake the recipes
## tfidf model
dta_tfidf <- tfidf_main %>%
  prep(model_data) %>%
  juice() 
dta_tfidf$ecthr_label <- as_factor(dta_tfidf$ecthr_label)
levels(dta_tfidf$ecthr_label)
## glove model
dta_glove <- glove_main %>%
  prep(model_data) %>%
  juice()
dta_glove$ecthr_label <- as_factor(dta_glove$ecthr_label)
levels(dta_glove$ecthr_label)

### set up the tasks
## tfidf model
tfidf_task <-TaskSupervised$new(
  id = "tfidf_model", 
  task_type = "classif", 
  backend = dta_tfidf,
  target = "ecthr_label"
  )
## glove model
glove_task <-TaskSupervised$new(
  id = "glove_model", 
  task_type = "classif", 
  backend = dta_tfidf,
  target = "ecthr_label"
)
## imbalance 
table(tfidf_task$truth())
tfidf_task$col_info
glove_task$col_info

### costs matrix
## the highest weight to false negatives, followed by lower costs for false positive
## imbalanced
costs <- matrix(c(0, 1, 1, 0), nrow = 2)
dimnames(costs) <- list(prediction = c("0", "1"), truth = c("0", "1"))
print(costs)
## adjusted - false negatives are 5 times more costly than false positives
costs <- matrix(c(0, 1, 5, 0), nrow = 2)
dimnames(costs) <- list(prediction = c("0", "1"), truth = c("0", "1"))
print(costs)
### const-sensitive measure
## calculates the costs based on our cost matrix
cost_measure = msr("classif.costs", costs = costs)
print(cost_measure)

## We want to penalize more false negatives, then false positives

  rownames(costs) <- getTaskClassLevels(ecthr_task)
costs