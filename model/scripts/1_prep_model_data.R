#### setting things up ------------------------------------------------------------------
### Packs
require(tidyverse)
require(lubridate)
### Dirs
parent_dir <- '/home/jmr/Dropbox/Current projects/thesis_papers/transparency, media, and compliance with HR Rulings/ecthr_media&compliance/data/media_data/3_classify_ecthr_news' 
rulart_dyad_path <- paste(parent_dir, "features", "data", "interm_data", "rulings_article_dyad_data_raw.csv.gz", sep = "/")
articles_data_path <- paste(parent_dir, "features", "data", "interm_data", "articles_data_raw.csv.gz", sep = "/")
tf_path <- paste(parent_dir, "features", "data", "input", "tf_dtm", sep = "/")
stringdist_path <- paste(parent_dir, "features", "data", "input", "stringdist", sep = "/")

#### String distance ---------------------------------------------------------------------
### load and concatenate the string distance data
sd_list <- list.files(stringdist_path, full.names = TRUE) 
## load
stringdist <- map_df(sd_list, read_csv) %>%
  select(-c(1, ruling_doc_file)) # keep only case id and article id as keys

#### Date distance, article nchr and source lang ------------------------------------------------------------------------
### load the articles data
arts_all <- read_csv(articles_data_path) %>%
  mutate(article_nchar = nchar(text), ## fill in missing date published
         date_published = replace(date_published, which(is.na(date_published)), ymd(str_extract(article_id, "(?<=\\_)[0-9]{4}\\-[0-9]{1,2}\\-[0-9]{1,2}(?=\\_)"))),
         date_distance = as.integer(judgment_date - date_published)) %>%
  select(ecthr_label, article_id, case_id, text, article_nchar, source_lang_alpha2, date_distance, date_published, judgment_date) # filter the relevant vars

#### Make the final dataset ----------------------------------------------------------------------
### combine
dataset_all <- tidy_dtm %>%
  left_join(arts_all) %>%
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

