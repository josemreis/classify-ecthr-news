
best_k <- function(k) {
  
  tfidf_main <- recipe(
    ecthr_label ~ .,
    data = train_data,
  ) %>%
    update_role(model_id, new_role = "id variable") %>%
    step_rm(c(lang_og, model_id, jaccard_distance, country_match_ratio, contains("ner"))) %>% ## remove the language variable (see above)
    step_mutate(## discretize the date distance variable
      date_distance = date_distance %>% str_replace("-", "minus_") %>% as_factor()
    ) %>%
    step_textfeature(text) %>%
    step_dummy(date_distance) %>%
    step_log(article_nchar) %>%
    step_tokenize(text) %>%
    step_word_embeddings(text, embeddings = textdata::embedding_glove6b(dimensions = 50)) %>%
    step_zv(all_predictors()) %>%
    step_normalize(all_numeric(), -article_nchar, -all_outcomes()) %>%
    step_smote(ecthr_label, seed = 1234, neighbors = k) %>%
    prep(train_data)
  
  train <- tfidf_main  %>%
    juice() 
  
  test <- tfidf_main %>%
    bake(test_data) 
  
  ### Cost-sensitive random forests
  rf_cs <- rand_forest(
    trees = 1000, # number of randomly samped predictor in each split when creating the trees
  ) %>%
    set_mode("classification") %>%
    set_engine(
      engine = "ranger",
      seed = 1234,
      num.threads = 4,
      importance = "impurity",
      sample.fraction = c(.5, .5)
    )
  
  rf_model <- rf_cs %>%
    fit(ecthr_label ~ ., data = train)
  
  eval_tibble <- test %>%
    select(ecthr_label) %>%
    mutate(
      class = parsnip:::predict_class(rf_model, test)
    )
  
  
  pred <- predict(rf_model, test, type = "prob")
  
  
  conf_mat(eval_tibble, ecthr_label, class)
  bal_accuracy(eval_tibble, ecthr_label,class)
  (kap <- kap(eval_tibble, ecthr_label,class))
  (f <- f_meas(eval_tibble, ecthr_label,class, event_level = "second"))
  
  out <- rbind(kap, f) %>%
    mutate(neighbor = k)
  
  return(out)
  
}

k_df <- map_df(seq(8, 32, by = 1), best_k)
k_df %>% 
  ggplot(aes(neighbor, .estimate, color = .metric)) + 
  geom_point() + 
  geom_line() +
  ggtitle("Comparing sizes of K for smote sampling") + 
  labs(caption = "ranger with default parameters")
