#### Setting things up ----------------------------------------------------------------------
### packs
require(tidyverse)
require(tidymodels)
require(stacks)

### wd
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

#### Model stacking -------------------------------------------------------------------------
### load the workflows
lasso_wf <- read_rds("models/wv-lasso_workflow.Rds") %>%
  pluck(".workflow", 1)

rf_wf <- read_rds("models/rf-wv_workflow.Rds") %>%
  pluck(".workflow", 1)

# svmrbf_wf <- read_rds("models/svmrbf-wv_workflow.Rds") %>%
#   pluck(".workflow", 1)

### load the tunning metrics
## select top 10 by Kappa
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

### fit the sub-models
## Prepare the folds
set.seed(1234)
folds <- rsample::vfold_cv(train_data, v = 10, repeats = 3)
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

# doParallel::registerDoParallel()
# tune_svmrbf <- tune_grid(
#   svmrbf_wf,
#   resamples = folds,
#   control = control_resamples(save_pred = TRUE, save_workflow = TRUE),
#   metrics = metric_set(kap, bal_accuracy, roc_auc, f_meas, ppv, npv, recall, precision)
# )

### stacking
# initialize the stack
model_stack <- stacks() %>% 
  # add candidate members
  stack_add(tune_lasso) %>%
  stack_add(tune_rf) %>%
  # determine how to combine their predictions
  stack_blend() 

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
  mutate(.pred_class = if_else(.pred_1 > 0.5, "1", "0") %>% factor(levels = c("1", "0"))) %>%
  conf_mat(ecthr_label, .pred_class)
