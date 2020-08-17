  #### Setting things up ----------------------------------------------------------------------
  ### packs
  require(tidyverse)
  require(tidymodels)
  remotes::install_github("tidymodels/stacks", ref = "main")
  require(stacks)
  
  ### wd
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
  
  #### Model stacking -------------------------------------------------------------------------
  ### load the workflows
  lasso_wf <- read_rds("models/wv-lasso_workflow.Rds") %>%
    pluck(".workflow", 1)
  rf_wf <- read_rds("models/rf-wv_workflow.Rds") %>%
    pluck(".workflow", 1)
  svmrbf_wf <- read_rds("models/svmrbf-wv_model.Rds")
  xgb_wf <- read_rds("models/xgb-wv_model.Rds")
  
  ### load the tunning metrics
  ## select top 20 by Kappa
  lasso_metrics <- read_csv("models/metrics/wv-lasso_metrics_tune.csv") %>%
    filter(.metric == "kap") %>%
    arrange(desc(mean)) %>%
    slice(1:20) %>%
    distinct(penalty, mixture, .keep_all = TRUE)
  
  rf_metrics <- read_csv("models/metrics/rf-wv_metrics_tune.csv") %>%
    filter(.metric == "kap") %>%
    arrange(desc(mean)) %>%
    slice(1:20) %>%
    distinct(mtry, min_n, .keep_all = TRUE)
  
  svmrbf_metrics <- read_csv("models/metrics/svmrbf_metrics_tune.csv") %>%
    filter(.metric == "kap") %>%
    arrange(desc(mean)) %>%
    slice(1:20) %>%
    distinct(cost, rbf_sigma, .keep_all = TRUE)
  
  xgb_metrics <- read_csv("models/metrics/xgb_metrics_tune.csv") %>%
    filter(.metric == "kap") %>%
    arrange(desc(mean)) %>%
    slice(1:20) %>%
    distinct(mtry, min_n, tree_depth, loss_reduction, sample_size, .keep_all = TRUE)
  
  ### prepare the hyper-parameter 
  ##  hyperparameters bounded by the lowest and highest models' parameters (top-20)
  set.seed(1234)
  grid_lasso <- grid_regular(
    penalty(range = c(
      lasso_metrics$penalty[which.min(lasso_metrics$penalty)],
      lasso_metrics$penalty[which.max(lasso_metrics$penalty)]
    ),
    trans = NULL),
    mixture(range = c(
      lasso_metrics$mixture[which.min(lasso_metrics$mixture)],
      lasso_metrics$mixture[which.max(lasso_metrics$mixture)]
    )),
    levels = 5
  )
  
  set.seed(1234)
  grid_rf <- grid_regular(
    mtry(range = c(
      rf_metrics$mtry[which.min(rf_metrics$mtry)],
      rf_metrics$mtry[which.max(rf_metrics$mtry)]
    )),
    min_n(range = c(
      rf_metrics$min_n[which.min(rf_metrics$min_n)],
      rf_metrics$min_n[which.max(rf_metrics$min_n)]
    )),
    levels = 5
  )
  
  set.seed(1234)
  grid_svmrbf <- grid_regular(
    cost(range = c(
      svmrbf_metrics$cost[which.min(svmrbf_metrics$cost)],
      svmrbf_metrics$cost[which.max(svmrbf_metrics$cost)]
    ),
    trans = NULL),
    rbf_sigma(range = c(
      svmrbf_metrics$rbf_sigma[which.min(svmrbf_metrics$rbf_sigma)],
      svmrbf_metrics$rbf_sigma[which.max(svmrbf_metrics$rbf_sigma)]
    ),
    trans = NULL),
    levels = 20
  )
  
  set.seed(1234)
  ## grid expansion would be to large
  grid_xgb <- grid_max_entropy(
    mtry(range = c(
      xgb_metrics$mtry[which.min(xgb_metrics$mtry)],
      xgb_metrics$mtry[which.max(xgb_metrics$mtry)]
    )),
    min_n(range = c(
      xgb_metrics$min_n[which.min(xgb_metrics$min_n)],
      xgb_metrics$min_n[which.max(xgb_metrics$min_n)]
    )),
    tree_depth(range = c(
      xgb_metrics$tree_depth[which.min(xgb_metrics$tree_depth)],
      xgb_metrics$tree_depth[which.max(xgb_metrics$tree_depth)]
    )),
    sample_prop(range = c(
      xgb_metrics$sample_size[which.min(xgb_metrics$sample_size)] %>% round(2),
      xgb_metrics$sample_size[which.max(xgb_metrics$sample_size)] %>% round(2)
    )),
    loss_reduction(range = c(
      xgb_metrics$loss_reduction[which.min(xgb_metrics$loss_reduction)] %>% round(2),
      xgb_metrics$loss_reduction[which.max(xgb_metrics$loss_reduction)] %>% round(2)
    )),
    learn_rate(range = c(
      xgb_metrics$learn_rate[which.min(xgb_metrics$learn_rate)] %>% round(2),
      xgb_metrics$learn_rate[which.max(xgb_metrics$learn_rate)] %>% round(2)
    )),
    size = 400
  )
  
  
  ### fit the sub-models
  ## Prepare the folds
  set.seed(1234)
  folds <- rsample::vfold_cv(train_data, v = 5, repeats = 3)
  ## fit
  doParallel::registerDoParallel()
  tune_lasso <- tune_grid(
    object = lasso_wf %>% update_model(
      logistic_reg(penalty = tune(), mixture = tune()) %>%
        set_mode("classification") %>%
        set_engine("glmnet")
    ),
    resamples = folds,
    grid = grid_lasso,
    control = control_resamples(save_pred = TRUE, save_workflow = TRUE, verbose = TRUE),
    metrics = metric_set(kap, bal_accuracy, roc_auc, f_meas, ppv, npv, recall, precision)
  )
  
  doParallel::registerDoParallel()
  tune_rf <- tune_grid(
    object = rf_wf %>% update_model(
      rand_forest(
        mtry = tune(),
        min_n = tune(),
        trees = 500
      ) %>%
        set_mode("classification") %>%
        set_engine(
          engine = "ranger",
          seed = 1234,
          num.threads = 4,
          importance = "impurity",
          sample.fraction = c(.5, .5)
        )
    ),
    resamples = folds,
    grid = grid_rf,
    control = control_resamples(save_pred = TRUE, save_workflow = TRUE, verbose = TRUE),
    metrics = metric_set(kap, bal_accuracy, roc_auc, f_meas, ppv, npv, recall, precision)
  )
    
    
    doParallel::registerDoParallel()
    tune_svmrbf <- tune_grid(
      object = svmrbf_wf %>% update_model(
        svm_rbf(
          cost = tune(),
          rbf_sigma = tune()
        ) %>%
          set_mode("classification") %>%
          set_engine(engine = "kernlab")
      ),
      resamples = folds,
      grid = grid_svmrbf,
      control = control_resamples(save_pred = TRUE, save_workflow = TRUE, verbose = TRUE),
      metrics = metric_set(kap, bal_accuracy, roc_auc, f_meas, ppv, npv, recall, precision)
    )
    
    doParallel::registerDoParallel()
    tune_xgb <- tune_grid(
      object = xgb_wf %>% update_model(
        boost_tree(
          trees = 1000, 
          tree_depth = tune(), min_n = tune(),
          loss_reduction = tune(),  
          sample_size = tune(), mtry = tune(),         
          learn_rate = tune(),                    
        ) %>% 
          set_engine("xgboost") %>% 
          set_mode("classification")
      ),
      resamples = folds,
      grid = grid_xgb,
      control = control_resamples(save_pred = TRUE, save_workflow = TRUE, verbose = TRUE),
      metrics = metric_set(kap, bal_accuracy, roc_auc, f_meas, ppv, npv, recall, precision)
    )
    
### stacking
# initialize the stack
data_stack <- stacks() %>% 
  # add candidate members
  stack_add(tune_lasso) %>%
  stack_add(tune_rf) %>%
  stack_add(tune_svmrbf) %>%
  stack_add(tune_xgb) 

### Stacks is a work in progress. Found a bug in stack_blend. Fixed it and wrote up my own.
my_blend <- function(data_stack = NULL, penalty = 10^(-6:-1), verbose = TRUE) {
  
  stacks:::check_inherits(data_stack, "data_stack")
  stacks:::check_blend_data_stack(data_stack)
  stacks:::check_penalty(penalty)
  stacks:::check_inherits(verbose, "logical")
  outcome <- attr(data_stack, "outcome")
  preds_formula <- paste0(outcome, " ~ .") %>% as.formula()
  lvls <- levels(data_stack[[outcome]])
  dat <- tibble::as_tibble(data_stack)
  if (attr(data_stack, "mode") == "regression") {
    model_spec <- parsnip::linear_reg(penalty = tune::tune(), 
                                      mixture = 1) %>% parsnip::set_engine("glmnet", lower.limits = 0)
    metric <- yardstick::metric_set(yardstick::rmse)
    preds_wf <- workflows::workflow() %>% workflows::add_model(model_spec) %>% 
      workflows::add_formula(preds_formula)
  } else {
    col_filter <- paste0(".pred_", lvls[1])
    dat <- dat %>% dplyr::select(-dplyr::starts_with(!!col_filter))
    if (length(lvls) == 2) {
      model_spec <- parsnip::logistic_reg(penalty = tune::tune(), 
                                          mixture = 1) %>% parsnip::set_engine("glmnet", 
                                                                               lower.limits = 0) %>% parsnip::set_mode("classification")
    } else {
      model_spec <- parsnip::multinom_reg(penalty = tune::tune(), 
                                          mixture = 1) %>% parsnip::set_engine("glmnet", 
                                                                               lower.limits = 0) %>% parsnip::set_mode("classification")
    }
    ### bug fix, if collumn contains a list, instead of double, average the predictions
    dat_fixed <- dat %>%
      mutate(
        across(
          where(is.list),
          ~ .x %>% unlist() %>% as.vector() %>% mean(.)
        )
      )
    metric <- yardstick::metric_set(yardstick::roc_auc)
    preds_wf <- workflows::workflow() %>% workflows::add_recipe(recipes::recipe(preds_formula, 
                                                                                data = dat_fixed)) %>% workflows::add_model(model_spec)
  }
  
  get_models <- function(x) {
    x %>% workflows::pull_workflow_fit() %>% purrr::pluck("fit")
  }
  
  splits <- attr(data_stack, "splits")
  
  if (inherits(splits[[1]], "val_split")) {
    
    rs <- rsample::bootstraps(dat, times = 20)
    
  } else {
    rs <- stacks:::reconstruct_resamples(attr(data_stack, "splits"), 
                                         dat_fixed)
  }
  candidates <- preds_wf %>% tune::tune_grid(resamples = rs, 
                                             grid = tibble::tibble(penalty = penalty), metrics = metric, 
                                             control = tune::control_grid(save_pred = TRUE, extract = get_models))
  coefs <- model_spec %>% tune::finalize_model(tune::select_best(candidates)) %>% 
    generics::fit(formula = preds_formula, data = dat_fixed)
  model_stack <- structure(list(model_defs = attr(data_stack, 
                                                  "model_defs"), coefs = coefs, metrics = stacks:::glmnet_metrics(candidates), 
                                equations = stacks:::get_expressions(coefs), cols_map = attr(data_stack, 
                                                                                             "cols_map"), model_metrics = attr(data_stack, "model_metrics"), 
                                train = attr(data_stack, "train"), mode = attr(data_stack, 
                                                                               "mode"), outcome = attr(data_stack, "outcome"), data_stack = dat_fixed, 
                                splits = attr(data_stack, "splits")), class = c("linear_stack", 
                                                                                "model_stack", "list"))
  if (stacks:::model_stack_constr(model_stack)) {
    model_stack
  }
}

model_stack <- my_blend(data_stack = data_stack)  

## hacky fix for the missing ".config" bug
# model_stack[["model_metrics"]] <- map(model_stack[["model_metrics"]], ~.x %>%
#                                         dplyr::mutate(.config = ""))
  
# fit
stack_mod <- model_stack %>%
  stack_fit(verbose = TRUE)

### Predict
ens_pred <- test_data %>%
  bind_cols(predict(stack_mod, ., type = "prob"))

ens_pred %>%
  mutate(.pred_class = if_else(.pred_1 > 0.4, "1", "0") %>% factor(levels = c("1", "0"))) %>%
  conf_mat(ecthr_label, .pred_class)

ens_pred %>%
  mutate(.pred_class = if_else(.pred_1 > 0.45, "1", "0") %>% factor(levels = c("1", "0"))) %>%
  kap(ecthr_label, .pred_class)
