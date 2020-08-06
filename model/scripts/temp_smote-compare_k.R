## load the workflow
best_fit <- read_rds("models/rf-glove_workflow.Rds")
best_wf <- best_fit %>%
  pluck(".workflow", 1)

best_k <- function(k) {
  
  ## update recipe
  new_recipe <- recipe(
    ecthr_label ~ .,
    data = train_data,
  ) %>%
    update_role(model_id, new_role = "id variable") %>% ## removing several variables which reduced/neutral_to performance
    step_rm(c(lang_og, model_id, jaccard_distance, country_match_ratio, contains("ner"))) %>% ## remove the language variable (see above)
    step_mutate(## discretize the date distance variable
      date_distance = date_distance %>% str_replace("-", "minus_") %>% as_factor()
    ) %>%
    step_dummy(date_distance) %>% ## date as dummy
    step_log(article_nchar) %>% ## log numbe rof characets
    step_tokenize(text) %>% ## tokenize and vectorize
    step_word_embeddings(text, embeddings = embeddings) %>%
    step_zv(all_predictors()) %>% # remove variables with zero variance
    step_normalize(all_numeric(), -article_nchar, -all_outcomes()) %>% ## center and scale
    themis::step_smote(ecthr_label, neighbors = k, seed = 1234)
  
  ## new workflow
  new_wf <- best_wf %>%
    update_recipe(new_recipe)
  # fit
  fit2 <- new_wf %>%
    last_fit(the_split, metrics = metric_set(roc_auc, bal_accuracy, f_meas, ppv, npv, recall, precision, kap))
  # generate predictions from the test set
  test_predictions <- fit2 %>% collect_predictions()
  ## confusion matrix
  test_predictions %>%
    conf_mat(ecthr_label, .pred_class) %>%
    print()
  
  ## assess
  test_performance <- fit2 %>% 
    collect_metrics() %>%
    mutate(model = "ranger",
           neighbors = k)
  
  print(test_performance)
  return(test_performance)
  
  
}
    
    k_df <- furrr::future_map_dfr(seq(1, 32, by = 1), best_k)
    k_df %>% 
      ggplot(aes(neighbors, .estimate, color = .metric)) + 
      geom_point() + 
      geom_line() +
      ggtitle("Comparing sizes of K for smote sampling") + 
      labs(caption = "ranger with default parameters")
    
    
    bk_df %>%
      filter(.metric == "f_meas") %>%
      arrange(desc(.estimate)) 
    
    k_df %>%
      filter(.metric == "f_meas") %>%
      arrange(desc(.estimate))
    

###  Best k for border-smote
## load the workflow
best_fit <- read_rds("models/rf-glove_workflow.Rds")
best_wf <- best_fit %>%
  pluck(".workflow", 1)
## function
best_bk <- function(k) {
  
  ## update recipe
  new_recipe <- recipe(
    ecthr_label ~ .,
    data = train_data,
  ) %>%
    update_role(model_id, new_role = "id variable") %>% ## removing several variables which reduced/neutral_to performance
    step_rm(c(lang_og, model_id, jaccard_distance, country_match_ratio, contains("ner"))) %>% ## remove the language variable (see above)
    step_mutate(## discretize the date distance variable
      date_distance = date_distance %>% str_replace("-", "minus_") %>% as_factor()
    ) %>%
    step_dummy(date_distance) %>% ## date as dummy
    step_log(article_nchar) %>% ## log numbe rof characets
    step_tokenize(text) %>% ## tokenize and vectorize
    step_word_embeddings(text, embeddings = embeddings) %>%
    step_zv(all_predictors()) %>% # remove variables with zero variance
    step_normalize(all_numeric(), -article_nchar, -all_outcomes()) %>% ## center and scale
    themis::step_bsmote(ecthr_label, neighbors = k, seed = 1234)
  
  ## new workflow
  new_wf <- best_wf %>%
    update_recipe(new_recipe)
  # fit
  fit2 <- new_wf %>%
    last_fit(the_split, metrics = metric_set(roc_auc, bal_accuracy, f_meas, ppv, npv, recall, precision, kap))
  # generate predictions from the test set
  test_predictions <- fit2 %>% collect_predictions()
  ## confusion matrix
  test_predictions %>%
    conf_mat(ecthr_label, .pred_class) %>%
    print()
  
  ## assess
  test_performance <- fit2 %>% 
    collect_metrics() %>%
    mutate(model = "ranger",
           neighbors = k)
  
  print(test_performance)
  return(test_performance)
  
}

bk_df <- map_df(seq(8, 32, by = 1), best_bk)
bk_df %>% 
  ggplot(aes(neighbors, .estimate, color = .metric)) + 
  geom_point() + 
  geom_line() +
  ggtitle("Comparing sizes of K for border smote sampling") + 
  labs(caption = "ranger")

bk_df %>% 
  mutate(sampling_technique = "smote") %>%
  bind_rows(
    k_df %>%
      mutate(sampling_technique = "border smote")
  ) %>%
  ggplot(aes(neighbors, .estimate, color = .metric)) + 
  geom_point() + 
  geom_line() +
  facet_wrap(~sampling_technique) +
  theme_minimal() +
  ggtitle("Comparing sizes of K for smote sampling") + 
  labs(caption = "ranger")


bk_df %>% 
  mutate(sampling_technique = "smote") %>%
  bind_rows(
    k_df %>%
      mutate(sampling_technique = "border smote")
  ) %>%
  filter(.metric == "f_meas") %>%
  arrange(desc(.estimate)) %>%
  View()


