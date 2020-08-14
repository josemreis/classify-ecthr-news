#### setting things up ------------------------------------------------------------------
### Packs
require(tidyverse)
require(lubridate)
require(spacyr)
require(furrr)

### Dirs
parent_dir <- '/home/jmr/Dropbox/Current projects/thesis_papers/transparency, media, and compliance with HR Rulings/ecthr_media&compliance/data/media_data/3_classify_ecthr_news' 
rulart_dyad_path <- paste(parent_dir, "features", "data", "interm_data", "rulings_article_dyad_data_raw.csv.gz", sep = "/")
articles_data_path <- paste(parent_dir, "features", "data", "interm_data", "articles_data_raw.csv.gz", sep = "/")
tf_path <- paste(parent_dir, "features", "data", "input", "tf_dtm", sep = "/")
stringdist_path <- paste(parent_dir, "features", "data", "input", "stringdist", sep = "/")

## Update the labels
update_labels <- FALSE
## source the python script
if (update_labels) {
  
  reticulate::source_python('/home/jmr/Dropbox/Current projects/thesis_papers/transparency, media, and compliance with HR Rulings/ecthr_media&compliance/data/media_data/3_classify_ecthr_news/features/scripts/generate_text_features.py')
  
}

#### String distance ---------------------------------------------------------------------
### load and concatenate the string distance data
sd_list <- list.files(stringdist_path, full.names = TRUE) 
## load
stringdist <- map_df(sd_list, read_csv) %>%
  select(-c(1, ruling_doc_file)) # keep only case id and article id as keys

#### Date distance, article nchr and source lang ------------------------------------------------------------------------
### load the articles data
arts_all <- read_csv(articles_data_path)

## fill in missing dates for denmark and belgium
to_fill <- read_csv('/home/jmr/Dropbox/Current projects/thesis_papers/transparency, media, and compliance with HR Rulings/ecthr_media&compliance/data/media_data/1_data_collection/alternative_sources/scrape/country-level/Belgium/data/2_ecthrMatch&counts_added.csv.gz') %>%
  select(article_id, date_published2 = published_date) %>%
  rbind(
    read_csv('/home/jmr/Dropbox/Current projects/thesis_papers/transparency, media, and compliance with HR Rulings/ecthr_media&compliance/data/media_data/1_data_collection/alternative_sources/scrape/country-level/Denmark/data/2_ecthrMatch&counts_added.csv.gz') %>%
      select(article_id, date_published2 = published_date)
  ) %>% 
  mutate(date_published2 = as_date(ymd_hms(date_published2)))

## prep the arts data
arts_data <- left_join(arts_all, to_fill) %>% ## fill in missing
  mutate(date_published = if_else(is.na(date_published) & !is.na(date_published2),
                                  date_published2,
                                  date_published), ## generate the other features
         date_published = ymd(date_published)) %>%
  mutate(article_nchar = nchar(text),
         date_distance = as.integer(date_published - judgment_date)) %>%
  select(ecthr_label, article_id, case_id, text, article_nchar, source_lang_alpha2, date_distance, date_published, judgment_date)

#### NER variables -----------------------------------------------------------------------------------------
### From spacy, extract all named entities

if (!file.exists(paste(parent_dir, "model", "data", "interm_data", "ner_data.csv.gz", sep = "/"))) {
  
# initialize spacy
spacyr::spacy_initialize(python_executable = '/home/jmr/anaconda3/bin/python')
## turn text and id to a named vector
docs <- arts_data$text %>%
  set_names(arts_data$article_id)
## parse
parsed <- spacy_parse(
  x = docs, 
  lemma = FALSE, 
  entity = TRUE, 
  pos = FALSE, 
  multithread = TRUE
)
## export
just_ner <- parsed %>%
  rename(article_id = doc_id) %>%
  group_by(article_id) %>% 
  mutate(token_n = n()) %>% 
  ungroup() %>% 
  filter(nchar(entity) > 0) %>% 
  group_by(article_id) %>% 
  mutate(entity_n = n()) %>% 
  ungroup()

write_csv(just_ner,
          path = gzfile(paste(parent_dir, "model", "data", "interm_data", "ner_data.csv.gz", sep = "/")))

spacyr::spacy_finalize()

} else {
  
  just_ner <- read_csv(paste(parent_dir, "model", "data", "interm_data", "ner_data.csv.gz", sep = "/"))
  
}

### first batch of vars: NER counts normalized by number of words
ner_counts <- just_ner %>%
  select(article_id, entity, token_n, entity_n) %>%
  mutate(entity = str_remove(entity, "\\_.*?$"),
         entity = str_to_lower(entity)) %>%
  count(article_id, token_n, entity_n, entity) %>%
  mutate(entity_count_norm = n/token_n) %>%
  pivot_wider(
    names_from = "entity", 
    names_prefix = "ner_", 
    values_from = "entity_count_norm", 
    values_fill = 0
    ) %>%
  select(-token_n, -entity_n) %>%
  distinct(article_id, .keep_all = TRUE)

## join
nercounts_added <- arts_data %>%
  left_join(ner_counts)

### country mention ratio -------------------------------------------------------------------------------------------
## extract the mentions to countries, standardize them, make a ratio of countries to case country
## ner countries
ner_countries <- just_ner %>%
  filter(str_detect(entity, "GPE")) %>%
  left_join(arts_data %>%
              select(article_id, source_lang_alpha2)) %>%
  mutate(match_country = future_map_chr(token, ~countrycode::countrycode(.x, origin = 'country.name', destination = 'iso2c')),
         match_country = match_country %>% str_to_lower()) %>%
  mutate(country_match_ratio = source_lang_alpha2 == match_country) %>%
  group_by(article_id) %>%
  summarise(country_match_ratio = mean(country_match_ratio, na.rm = TRUE),
            country_match_ratio = ifelse(is.nan(country_match_ratio),
                                         0, 
                                         country_match_ratio)) %>%
  ungroup()

## add it
countrymatch_added <- nercounts_added %>%
  left_join(ner_countries) %>%
  mutate(country_match_ratio = ifelse(is.na(country_match_ratio),
                                      0, 
                                      country_match_ratio))

#### Verbs related vars----------------------------------------------------------------------------------------------
### given that the issue is time-overlap between rulings for the same countries it is reasonable to encode information about tenses of the verbs
## Proportion of the following vars:
# modal vebs ("shall" or "will"): 'prop_md'
# past tense: 'prop_vbd'
# past parciple: 'prop_vbn'
# present: 'prop_vbp'
# infinitive: 'prop_vb'

if (!file.exists(paste(parent_dir, "model", "data", "interm_data", "verbs_data.csv.gz", sep = "/"))) {
  
  #initialize spacy
  spacyr::spacy_initialize(python_executable = '/home/jmr/anaconda3/bin/python')
  ## turn text and id to a named vector
  docs <- arts_data$text %>%
    set_names(arts_data$article_id)
  
  ## parse
  parsed <- spacy_parse(
    x = docs, 
    pos = FALSE, 
    lemma = FALSE, 
    entity = FALSE, 
    tag = TRUE, 
    dependency = FALSE, 
    nounphrase = TRUE,
    multithread = TRUE
  )
  
  ## export
  just_verbs <- parsed %>%
    filter(tag %in% c("MD", "VBD", "VBN", "VB", "VBP", "VBZ", "VBG")) %>%
    rename(article_id = doc_id) %>%
    group_by(article_id) %>% 
    mutate(token_n = n()) %>% 
    ungroup() %>% 
    group_by(article_id) %>% 
    mutate(entity_n = n()) %>% 
    ungroup()
  
  write_csv(just_verbs,
            path = gzfile(paste(parent_dir, "model", "data", "interm_data", "verbs_data.csv.gz", sep = "/")))
  
  spacyr::spacy_finalize()
  
} else {
  
  just_verbs <- read_csv(paste(parent_dir, "model", "data", "interm_data", "verbs_data.csv.gz", sep = "/"))
  
}

### count verbs per article id
verb_counts <- just_verbs %>%
  select(article_id, tag) %>%
  mutate(tag = str_to_lower(tag)) %>%
  count(article_id, tag) %>%
  pivot_wider(
    names_from = "tag", 
    names_prefix = "verb_count_", 
    values_from = "n", 
    values_fill = 0
  ) %>%
  distinct(article_id, .keep_all = TRUE)

## join
verbcounts_added <- countrymatch_added %>%
  left_join(verb_counts) %>%
  mutate(across(.cols = c(contains("verb_count")), ~.x/article_nchar))

#### some relevant matching vars --------------------------------------------------------------
### match sentencing/ruling words


#### Make the final dataset ----------------------------------------------------------------------
### combine
dataset_all <- verbcounts_added %>%
  left_join(stringdist) 
### modeling data
model_data <- dataset_all %>%
  filter(!is.na(ecthr_label))
## export
write_csv(model_data,
          path = gzfile(paste(parent_dir, "model", "data", "model_data.csv.gz", sep = "/")))

### to_predict data
to_predict_data <- dataset_all %>%
  anti_join(model_data)
## export
write_csv(to_predict_data,
          path = gzfile(paste(parent_dir, "model", "data", "to_predict_data.csv.gz", sep = "/")))

