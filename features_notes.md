# Main features

  1. 1-5 ngrams (stemmed - ?, past-tense... -, stop-words removed) represented by their IDF or TF-IDF (?? see below); numerical
  2. Any match between country mentioned in the document id and countries in the article; dummy
  3. date distance; int
  4. document similarity; as in my working-paper "transparency in international law..."; int
  6. source (dummy var); some sources are expected to be more likely to cover echr; dummy 

## Sources for case description

  - **decisions text**:
    1. robust, but takes longer;
    2. Problem with the varying languages, though we can translate it;
    3. Plus: we already have these decisions at '/home/jmr/Dropbox/Current projects/thesis_papers/transparency, media, and compliance with HR Rulings/ecthr_media&compliance/data/ecthr_transparency_data/rulings_dir.7z';
  - **Scrape the press-releases**:
    1. Languages expected to be closes to the media;
    2. Still would have to scrape them, see above;
    3. Not clear whether we have at least one for each and how to systematically get them - though that is the easy part with hudoc;
    4. Languages; see above

## document similarity

We will compare the similarity between the decision and the news article.

### Distance algorithm

...

### Similarity feature selection

#### Top words

  1. For all parsed rulings turn them into a tf-idf BoW so as to punish frequent words and to give higher weight to idionsycratic ngrams;
  2. Next, for each ruling, select the  top-n words using tf-idf as a metric;
  3. Do the same for the news articles using the same n;
  4. Compute the similarity for each dyad - cosine similarity seems right;

 * Questions:
 - TF-IDF v. IDF? e.g. ECHR would get  a very low score, that is not necessarily good.


#### NER-based

  1. For each ruling and news article extract all named entities using the same NER algorithm;
  2. Compare the vectors of named entities to measure overlapp - jaccard distance maybe?

#### Full documents

  1. Extract the "FACTS" section of the ruling, turn it to a BoW (one of the above); turn news article into a BoW
  2. Compute the cosine similarity between the vectors


sources:
  - https://peerj.com/articles/cs-93/
  - http://martijnwieling.nl/files/Medvedeva-submitted.pdf
