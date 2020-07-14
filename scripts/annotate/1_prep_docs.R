#### Setting things up
#### ---------------------------------------------------------------------------
### packages
require(tidyverse)
require(lubridate)
require(data.table)
require(furrr)

#### Combine translations data ------------------------------------------------------------------------------
listed_files <- list.files('/home/jmr/Dropbox/Current projects/thesis_papers/transparency, media, and compliance with HR Rulings/ecthr_media&compliance/data/media_data/2_wrangle_corpus/data/interm_data/translated', full.names = TRUE)

## all in one
corpus_df <- map_df(listed_files, ~ read_csv(.x) %>%
                      mutate_all(as.character)) %>%
  select(-X1)

#### Add metadata ----------------------------------------------------------------------------------

### load the full corpus dataset
corpus_meta <- read_delim('/home/jmr/Dropbox/Current projects/thesis_papers/transparency, media, and compliance with HR Rulings/ecthr_media&compliance/data/media_data/2_wrangle_corpus/data/all_matches.tsv.gz', delim = "\t")

### left join
joined <- left_join(corpus_df, corpus_meta)

#### turn the data into doccano friendly structure and export -------------------------------------------------------------

## For csv, doccano seems to request: (i) target text to be called "text; (ii) label var (will create a fake one and erase later)
out <- joined %>%
  mutate(text = leading_paragraph_translated,
         label = "fooh")

## split it up and export
# relevant vars
to_code <- out %>%
  select(article_id, case_id, judgment_date, date_published, text, label) %>%
  distinct(leading_paragraph_translated, .keep_all = TRUE)

# reshuffle
df <- to_code %>%
  sample_n(size = nrow(.))
## split
n <- 500
nr <- nrow(df)
splited <- split(df, rep(1:ceiling(nr/n), each=n, length.out=nr))
## export
map2(c(1:length(splited)), splited, ~ write_csv(.y, 
                                                path = paste0('/home/jmr/Dropbox/Current projects/thesis_papers/transparency, media, and compliance with HR Rulings/ecthr_media&compliance/data/media_data/3_classify_ecthr_news/data/input_doccano/input_doccano_all_', .x ,'.csv')))

### export unlabeled news
write_csv(out,
          '/home/jmr/Dropbox/Current projects/thesis_papers/transparency, media, and compliance with HR Rulings/ecthr_media&compliance/data/media_data/3_classify_ecthr_news/data/unlabeled_news_all.csv')