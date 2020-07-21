#### Packs
require(tidyverse)

### Dirs
parent_dir <- '/home/jmr/Dropbox/Current projects/thesis_papers/transparency, media, and compliance with HR Rulings/ecthr_media&compliance/data/media_data/3_classify_ecthr_news' 
rulart_dyad_path <- paste(parent_dir, "features", "data", "interm_data", "rulings_article_dyad_data_raw.csv.gz", sep = "/")
articles_data_path <- paste(parent_dir, "features", "data", "interm_data", "articles_data_raw.csv.gz", sep = "/")
tf_path <- paste(parent_dir, "features", "data", "input", "tf_dtm", sep = "/")
stringdist_path <- paste(parent_dir, "features", "data", "input", "stringdist", sep = "/")