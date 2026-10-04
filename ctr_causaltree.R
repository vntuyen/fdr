# -*- coding: utf-8 -*-
# =============================================================================
# ctr_causaltree.R
# =============================================================================
# R back-end for the CTR (Causality-based Therapy Recommendation) baseline,
# called from baselines.recommend_ctr() once per training fold / OOD split.
#
# The modelling code is taken from the official CTR implementation
#   https://github.com/vntuyen/ctr  (CTR_models.R :: build_causal_tree_model,
#                                    predict_causal_effect_row)
# CTR: one causal tree per treatment plan
# (that plan vs. all others), split.Rule = "CT", cv.option = "CT",
# honest splitting + honest CV, no bucketing, 5-fold internal xval, cp = 0,
# minsize = 5, pruning at the CP with minimum cross-validated error, and a
# deterministic seed (seed + arm index) per arm.
#
# Usage:
#   Rscript ctr_causaltree.R <train.csv> <test.csv> <out.csv> <outcome_col> \
#                            <seed> <TP1,TP2,...>
# =============================================================================

suppressPackageStartupMessages({
  library(rpart)
  library(causalTree)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 6) {
  stop("Usage: Rscript ctr_causaltree.R <train.csv> <test.csv> <out.csv> <outcome_col> <seed> <TP1,TP2,...>")
}
train_file     <- args[[1]]
test_file      <- args[[2]]
out_file       <- args[[3]]
outcome_name   <- args[[4]]
seed           <- as.integer(args[[5]])
causal_factors <- strsplit(args[[6]], ",", fixed = TRUE)[[1]]

K            <- 5L    # internal x-validation folds (CTR: xval = K = 5)
MINSIZE      <- 5L    # CTR: minsize = 5L
MIN_PER_ARM  <- 2L    # need >= 2 treated and >= 2 control to estimate an effect

trainingData <- read.csv(train_file, check.names = FALSE)
testData     <- read.csv(test_file,  check.names = FALSE)

# ---- Helper from CTR: coerce treatment indicators to 0/1 --------------------
to01 <- function(x) {
  if (is.logical(x)) return(as.integer(x))
  if (is.factor(x))  return(as.integer(x) - 1L)
  if (is.numeric(x)) {
    ux <- sort(unique(na.omit(x)))
    if (identical(ux, c(0, 1))) return(as.integer(x))
    if (identical(ux, c(1, 2))) return(as.integer(x - 1L))
    return(as.integer(x != 0))
  }
  stop("Treatment must be coercible to numeric 0/1.")
}

trainingData[[outcome_name]] <- as.numeric(trainingData[[outcome_name]])
for (f in causal_factors) {
  stopifnot(f %in% names(trainingData))
  trainingData[[f]] <- to01(trainingData[[f]])
  if (f %in% names(testData)) testData[[f]] <- to01(testData[[f]])
}

# ---- Drop constant predictors (never the outcome or treatments) — as in CTR --
is_constant  <- vapply(trainingData, function(col) length(unique(na.omit(col))) <= 1, logical(1))
protect      <- names(trainingData) %in% c(outcome_name, causal_factors)
trainingData <- trainingData[, !(is_constant & !protect), drop = FALSE]

old_contr <- options(contrasts = c("contr.treatment", "contr.poly"))

# ---- Propensity model (fac ~ all covariates except outcome + fac) -----------
PROPENSITY_MODE <- tolower(Sys.getenv("CTR_PROPENSITY", "ctr"))   # "ctr" (default) or "marginal"

fit_propensity <- function(fac) {
  if (PROPENSITY_MODE == "marginal") {
    return(mean(trainingData[[fac]]))
  }
  rhs_vars <- setdiff(names(trainingData), c(outcome_name, fac))
  rhs      <- if (length(rhs_vars) > 0) paste0("`", rhs_vars, "`", collapse = " + ") else "1"
  frm_prop <- as.formula(paste0("`", fac, "` ~ ", rhs))
  p <- tryCatch({
    reg <- suppressWarnings(glm(frm_prop, family = binomial(link = "logit"),
                                data = trainingData, control = glm.control(maxit = 100)))
    as.numeric(reg$fitted.values)
  }, error = function(e) {
    message(sprintf("  [CTR] propensity fit failed for %s (%s); using marginal rate.", fac, conditionMessage(e)))
    rep(mean(trainingData[[fac]]), nrow(trainingData))
  })
  p
}

# ---- One causal tree per treatment plan (CTR::build_causal_tree_model) ------
effects <- data.frame(row.names = seq_len(nrow(testData)))

for (i in seq_along(causal_factors)) {
  fac <- causal_factors[[i]]
  tr  <- trainingData[[fac]]
  n_t <- sum(tr == 1)
  n_c <- sum(tr == 0)

  if (n_t < MIN_PER_ARM || n_c < MIN_PER_ARM) {
    message(sprintf("  [CTR] %s: %d treated / %d control in training -> effect not estimable (NA).", fac, n_t, n_c))
    effects[[fac]] <- NA_real_
    next
  }

  set.seed(seed + i)   # deterministic RNG per arm, as in CTR
  propensity_scores <- fit_propensity(fac)

  pred <- tryCatch({
    tree <- causalTree::causalTree(
      formula      = as.formula(paste0("`", outcome_name, "` ~ .")),
      data         = trainingData,
      treatment    = trainingData[[fac]],
      split.Rule   = "CT",
      cv.option    = "CT",
      split.Honest = TRUE,
      cv.Honest    = TRUE,
      split.Bucket = FALSE,
      xval         = K,
      cp           = 0,
      minsize      = MINSIZE,
      propensity   = propensity_scores
    )
    # Prune at the CP with minimum cross-validated error (CTR)
    opcp  <- tree$cptable[, "CP"][which.min(tree$cptable[, "xerror"])]
    opfit <- prune(tree, cp = opcp)
    as.numeric(predict(opfit, newdata = testData))
  }, error = function(e) {
    y <- trainingData[[outcome_name]]
    ate <- mean(y[tr == 1]) - mean(y[tr == 0])
    message(sprintf("  [CTR] causalTree failed for %s (%s); root-node effect %.4f used.", fac, conditionMessage(e), ate))
    rep(ate, nrow(testData))
  })

  effects[[fac]] <- pred
}

options(old_contr)
write.csv(effects, out_file, row.names = FALSE)
