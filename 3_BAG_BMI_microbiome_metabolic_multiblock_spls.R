
library(mixOmics)
library(readr)
library(dplyr)
library(matrixStats)
library(ggplot2)

# ----------------------------
# 0) File paths
# ----------------------------
metab_csv <- "H:/postdoc/UCLA_postdoc/aging_BMI/with_church_data/result/microbio/merged_metabolic_final.csv"
micro_csv <- "H:/postdoc/UCLA_postdoc/aging_BMI/with_church_data/result/microbio/merged_metagenomic_final.csv"
phen_csv  <- "H:/postdoc/UCLA_postdoc/aging_BMI/with_church_data/result/microbio/BAI_final.csv"

# ----------------------------
# 1) Helpers
# ----------------------------

# simple CLR for compositional microbiome (optional; comment out if already normalized)
clr <- function(X, add = 1e-6) {
  X <- as.matrix(X)
  X <- X + add
  logX <- log(X)
  gm <- rowMeans(logX)
  sweep(logX, 1, gm, FUN = "-")
}

nzv_filter <- function(df, tol = 1e-8) {
  keep <- apply(df, 2, function(x) sd(x, na.rm = TRUE) > tol)
  df[, keep, drop = FALSE]
}

# ----------------------------
# 2) Read data
# ----------------------------
metab <- read_csv(metab_csv, show_col_types = FALSE)
micro <- read_csv(micro_csv, show_col_types = FALSE)
micro$sub_id <- NULL
# NOTE: phen_csv is read with col_names = FALSE -- confirm the file has no
# header row, otherwise the header would be silently read in as data.
pheno <- read_csv(phen_csv, col_names = FALSE, show_col_types = FALSE)

Xlist <- list(
  micro = micro,       # e.g., CLR/VST taxa or pathways
  metab = metab        # e.g., log-transformed metabolites
)

# ----------------------------
# 3) Build multiset (blocks) & design
# ----------------------------
B <- length(Xlist)
# Design: weak X-X links, strong X-Y links
design <- matrix(0.1, nrow = B + 1, ncol = B + 1)
design[B + 1, ] <- 1
design[, B + 1] <- 1
diag(design) <- 0

list.keepX <- list(
  metab = unique(pmax(5, round(ncol(metab) * 0.08))),
  micro = unique(pmax(10, round(ncol(micro) * 0.03)))
)

Y <- as.numeric(pheno[[1]])
Y_mat <- matrix(Y, ncol = 1)

Xlist <- lapply(Xlist, function(X) {
  X <- as.data.frame(X)
  num_idx <- vapply(X, is.numeric, logical(1))
  as.matrix(X[, num_idx, drop = FALSE])
})

n <- nrow(Y_mat)
q <- ncol(Y_mat)
folds <- 5

# ----------------------------
# 4) 5-fold CV block.spls (per-fold zero-variance filter + keepX cap)
# ----------------------------
# Fixed a priori seed (not selected for CV performance -- see header note).
seed <- 1
set.seed(seed)
fold_id <- sample(rep(seq_len(folds), length.out = n))
yhat_all <- matrix(NA_real_, nrow = n, ncol = q)

metagene_weights <- matrix(0, nrow = dim(Xlist$micro)[2], ncol = folds,
                    dimnames = list(colnames(Xlist$micro), paste0("fold", 1:folds)))
metabolo_weights <- matrix(0, nrow = dim(Xlist$metab)[2], ncol = folds,
                           dimnames = list(colnames(Xlist$metab), paste0("fold", 1:folds)))

for (f in seq_len(folds)) {
  te <- which(fold_id == f)
  tr <- setdiff(seq_len(n), te)

  X_tr <- lapply(Xlist, function(X) X[tr, , drop = FALSE])
  X_te <- lapply(Xlist, function(X) X[te, , drop = FALSE])
  Y_tr <- Y_mat[tr, , drop = FALSE]

  # zero-variance filter fit on the training fold only
  kept_cols <- lapply(X_tr, function(Xb) {
    sds <- colSds(Xb, na.rm = TRUE)
    colnames(Xb)[!(is.na(sds) | sds == 0)]
  })
  for (b in names(X_tr)) {
    X_tr[[b]] <- X_tr[[b]][, kept_cols[[b]], drop = FALSE]
    X_te[[b]] <- X_te[[b]][, kept_cols[[b]], drop = FALSE]
  }

  # cap keepX this fold so it never exceeds remaining features
  keepX_fold <- setNames(vector("list", length(X_tr)), names(X_tr))
  for (b in names(X_tr)) {
    keepX_fold[[b]] <- pmin(list.keepX[[b]], ncol(X_tr[[b]]))
    if (any(keepX_fold[[b]] == 0))
      stop(sprintf("keepX for block '%s' is zero after filtering (fold %d).", b, f))
  }

  fit_tr <- block.spls(
    X = X_tr, Y = Y_tr,
    ncomp = 1,
    keepX = keepX_fold,
    design = design,
    scale = TRUE, mode = "regression"
  )
  metagene_weights[kept_cols[["micro"]], f] <- fit_tr$loadings$micro
  metabolo_weights[kept_cols[["metab"]], f] <- fit_tr$loadings$metab

  pr <- predict(fit_tr, newdata = X_te)$predict
  if (is.list(pr)) pr <- if ("Y" %in% names(pr)) pr[["Y"]] else pr[[1]]

  if (is.array(pr) && length(dim(pr)) == 3) {
    yhat_fold <- pr[, , 1, drop = FALSE]
    yhat_fold <- matrix(yhat_fold, nrow = length(te), ncol = q)
  } else if (is.matrix(pr)) {
    yhat_fold <- if (ncol(pr) == q) pr else pr[, 1, drop = FALSE]
  } else {
    yhat_fold <- matrix(as.numeric(pr), ncol = 1)
  }
  yhat_all[te, ] <- yhat_fold
}

yt <- Y_mat[, 1]; yp <- yhat_all[, 1]
out_corr <- suppressWarnings(cor(yt, yp, use = "pairwise"))
out_rmse <- sqrt(mean((yt - yp)^2, na.rm = TRUE))
cat("5-fold CV block.spls: cor =", round(out_corr, 3), "RMSE =", round(out_rmse, 3),
    "seed =", seed, "\n")

# ----------------------------
# 5) Aggregate per-fold feature weights
# ----------------------------
mean_w_metagene <- rowMeans(metagene_weights, na.rm = TRUE)
mean_w_metabolo <- rowMeans(metabolo_weights, na.rm = TRUE)

df_metagene <- data.frame(Feature = names(mean_w_metagene),
                          Weight = as.numeric(mean_w_metagene),
                          row.names = NULL)
df_metabolo <- data.frame(Feature = names(mean_w_metabolo),
                          Weight = as.numeric(mean_w_metabolo),
                          row.names = NULL)

# ----------------------------
# 6) Top-weighted feature plots
# ----------------------------
topN <- 20

up <- head(df_metabolo[order(-df_metabolo$Weight), c("Feature", "Weight")], topN)
down <- head(df_metabolo[order(df_metabolo$Weight), c("Feature", "Weight")], topN)

ggplot(up, aes(x = reorder(Feature, Weight), y = Weight)) +
  geom_segment(aes(xend = Feature, y = 0, yend = Weight)) +
  geom_point() +
  coord_flip() +
  labs(title = "Top positively weighted metabolites", x = "", y = "Weight") +
  theme_classic()

ggplot(down, aes(x = reorder(Feature, -Weight), y = Weight)) +
  geom_segment(aes(xend = Feature, y = 0, yend = Weight)) +
  geom_point() +
  coord_flip() +
  labs(title = "Top negatively weighted metabolites", x = "", y = "Weight") +
  theme_classic()

up <- head(df_metagene[order(-df_metagene$Weight), c("Feature", "Weight")], topN)
down <- head(df_metagene[order(df_metagene$Weight), c("Feature", "Weight")], topN)

ggplot(up, aes(x = reorder(Feature, Weight), y = Weight)) +
  geom_segment(aes(xend = Feature, y = 0, yend = Weight)) +
  geom_point() +
  coord_flip() +
  labs(title = "Top positively weighted species", x = "", y = "Weight") +
  theme_classic()

ggplot(down, aes(x = reorder(Feature, -Weight), y = Weight)) +
  geom_segment(aes(xend = Feature, y = 0, yend = Weight)) +
  geom_point() +
  coord_flip() +
  labs(title = "Top negatively weighted species", x = "", y = "Weight") +
  theme_classic()

# ----------------------------
# 7) Save results
# ----------------------------
write.csv(df_metabolo, "H:/postdoc/UCLA_postdoc/aging_BMI/with_church_data/result/microbio/Metabolomics/multiset_spls_BAI_metabo.csv")
write.csv(df_metagene, "H:/postdoc/UCLA_postdoc/aging_BMI/with_church_data/result/microbio/Metabolomics/multiset_spls_BAI_metage.csv")
