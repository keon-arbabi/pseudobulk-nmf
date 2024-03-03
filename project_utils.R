voomByGroup <- function(counts, group = NULL, design = NULL, lib.size = NULL,
                        dynamic = NULL, normalize.method = "none",
                        span = 0.5, save.plot = TRUE, print = TRUE,
                        plot = c("none", "all", "separate", "combine"),
                        col.lines = NULL,
                        pos.legend = c("inside", "outside", "none"),
                        fix.y.axis = FALSE, ...) {
  # 14 June 2017 (Last updated 6 May 2022)
  # Charity Law, Xueyi Dong and Yue You
  # Copied from the github repo
  # https://github.com/YOU-k/voomByGroup/blob/main/voomByGroup.R 
  # on May 22 2023.

  library(dplyr)  

  out <- list()
  if (is(counts, "DGEList")) {
    out$genes <- counts$genes
    out$targets <- counts$samples
    if(is.null(group))
      group <- counts$samples$group
    # if (is.null(design) && diff(range(as.numeric(counts$sample$group))) > 0)
    #   design <- model.matrix(~group, data = counts$samples)
    if (is.null(lib.size))
      lib.size <- with(counts$samples, lib.size * norm.factors)
    counts <- counts$counts
  }
  else {
    isExpressionSet <- suppressPackageStartupMessages(
      is(counts, "ExpressionSet"))
    if (isExpressionSet) {
      if (length(Biobase::fData(counts)))
        out$genes <- Biobase::fData(counts)
      if (length(Biobase::pData(counts)))
        out$targets <- Biobase::pData(counts)
      counts <- Biobase::exprs(counts)
    }
    else {
      counts <- as.matrix(counts)
    }
  }
  if (nrow(counts) < 2L)
    stop("Need at least two genes to fit a mean-variance trend")
  # Library size
  if(is.null(lib.size))
    lib.size <- colSums(counts)
  # Group
  if(is.null(group))
    group <- rep("Group1", ncol(counts))
  group <- as.factor(group)
  intgroup <- as.integer(group)
  levgroup <- levels(group)
  ngroups <- length(levgroup)
  # Design matrix
  if (is.null(design)) {
    design <- matrix(1L, ncol(counts), 1)
    rownames(design) <- colnames(counts)
    colnames(design) <- "GrandMean"
  }
  # Dynamic
  if (is.null(dynamic)) {
    dynamic <- rep(FALSE, ngroups)
  }
  # voom by group
  if(print)
    cat("Group:\n")
  E <- w <- counts
  xy <- line <- as.list(rep(NA, ngroups))
  names(xy) <- names(line) <- levgroup
  for (lev in 1L:ngroups) {
    if(print)
      cat(lev, levgroup[lev], "\n")
    i <- intgroup == lev
    countsi <- counts[, i]
    libsizei <- lib.size[i]
    designi <- design[i, , drop = FALSE]
    QR <- qr(designi)
    if(QR$rank<ncol(designi))
      designi <- designi[,QR$pivot[1L:QR$rank], drop = FALSE]
    if(ncol(designi)==ncol(countsi))
      designi <- matrix(1L, ncol(countsi), 1)
    voomi <- limma::voom(
      counts = countsi, design = designi, lib.size = libsizei,
      normalize.method = normalize.method, span = span, plot = FALSE,
      save.plot = TRUE, ...
    )
    E[, i] <- voomi$E
    w[, i] <- voomi$weights
    xy[[lev]] <- voomi$voom.xy
    line[[lev]] <- voomi$voom.line
  }
  #voom overall
  if (TRUE %in% dynamic){
    voom_all <- limma::voom(
      counts = counts, design = design, lib.size = lib.size,
      normalize.method = normalize.method, span = span, plot = FALSE,
      save.plot = TRUE, ...
    )
    E_all <- voom_all$E
    w_all <- voom_all$weights
    xy_all <- voom_all$voom.xy
    line_all <- voom_all$voom.line

    dge <- edgeR::DGEList(counts)
    disp <- edgeR::estimateCommonDisp(dge)
    disp_all <- disp$common
  }
  # Plot, can be "both", "none", "separate", or "combine"
  plot <- plot[1]
  if(plot!="none"){
    disp.group <- c()
    for (lev in levgroup) {
      dge.sub <- edgeR::DGEList(counts[,group == lev])
      disp <- edgeR::estimateCommonDisp(dge.sub)
      disp.group[lev] <- disp$common
    }
    if(plot %in% c("all", "separate")){
      if (fix.y.axis == TRUE) {
        yrange <- sapply(levgroup, function(lev){
          c(min(xy[[lev]]$y), max(xy[[lev]]$y))
        }, simplify = TRUE)
        yrange <- c(min(yrange[1,]) - 0.1, max(yrange[2,]) + 0.1)
      }
      for (lev in 1L:ngroups) {
        if (fix.y.axis == TRUE){
          plot(
            xy[[lev]], xlab = "log2( count size + 0.5 )", 
            ylab = "Sqrt( standard deviation )", pch = 16, cex = 0.25, 
            ylim = yrange)
        } else {
          plot(
            xy[[lev]], xlab = "log2( count size + 0.5 )", 
            ylab = "Sqrt( standard deviation )", pch = 16, cex = 0.25)
        }
        title(paste("voom: Mean-variance trend,", levgroup[lev]))
        lines(line[[lev]], col = "red")
        legend("topleft", bty="n", paste("BCV:", 
          round(sqrt(disp.group[lev]), 3)), text.col="red")
      }
    }

    if(plot %in% c("all", "combine")){
      if(is.null(col.lines))
        col.lines <- 1L:ngroups
      if(length(col.lines)<ngroups)
        col.lines <- rep(col.lines, ngroups)
      xrange <- unlist(lapply(line, `[[`, "x"))
      xrange <- c(min(xrange)-0.3, max(xrange)+0.3)
      yrange <- unlist(lapply(line, `[[`, "y"))
      yrange <- c(min(yrange)-0.1, max(yrange)+0.3)
      plot(
        1L,1L, type="n", ylim=yrange, xlim=xrange, 
        xlab = "log2( count size + 0.5 )", ylab = "Sqrt( standard deviation )")
      title("voom: Mean-variance trend")
      if (TRUE %in% dynamic){
        for (dy in which(dynamic)){
          line[[dy]] <- line_all
          disp.group[dy] <- disp_all
          levgroup[dy] <- paste0(levgroup[dy]," (all)")
        }

      }
      for (lev in 1L:ngroups)
        lines(line[[lev]], col=col.lines[lev], lwd=2)
      pos.legend <- pos.legend[1]
      disp.order <- order(disp.group, decreasing = TRUE)
      text.legend <- paste(levgroup, ", BCV: ", round(sqrt(disp.group), 3), sep="")
      if(pos.legend %in% c("inside", "outside")){
        if(pos.legend=="outside"){
          plot(
            1,1, type="n", yaxt="n", xaxt="n", ylab="", xlab="", 
            frame.plot=FALSE)
          legend("topleft", 
            text.col=col.lines[disp.order], text.legend[disp.order], bty="n")
        } else {
          legend("topright", 
            text.col=col.lines[disp.order], text.legend[disp.order], bty="n")
        }
      }
    }
  }
  # Output
  if (TRUE %in% dynamic){
    E[,intgroup %in% which(dynamic)] <- E_all[,intgroup %in% which(dynamic)]
    w[,intgroup %in% which(dynamic)] <- w_all[,intgroup %in% which(dynamic)]
  }
  out$E <- E
  out$weights <- w
  out$design <- design
  if(save.plot){
    out$voom.line <- line
    out$voom.xy <- xy
  }
  new("EList", out)
}

cbind.fill <- function(...) {
  # https://stackoverflow.com/a/7962286
  data_list <- list(...)
  max_length <- max(sapply(data_list, length))
  data_padded <- lapply(data_list, function(x) {
    c(x, rep(NA, max_length - length(x)))
    })
  as.data.frame(data_padded)
}

check_nonestimable <- function(design, verbose = F) {
  non_estimable <- limma:::nonEstimable(design)
  if (!is.null(non_estimable)) {
    design <- design[, !colnames(design) %in% non_estimable]
    if (verbose) message("Dropping colinear terms from design matrix: ",
                         paste(non_estimable, collapse = ", "))
  }
  design
}

de_pseudobulk <- function(input, meta, model, contrast, cell_type = NULL,
                          de_method = "limma", de_type = "voombygroup",
                          verbose = FALSE, return_interm = FALSE) {

  #' @param input A list or matrix. For a list, it should consist of cell types 
  #' with lists of expr and meta generated by `Libra::to_pseudobulk()`. 
  #' `rownames(pseudobulks$meta)` must match `colnames(pseudobulks$expr)`. For a 
  #' matrix, it should be a count matrix of gene x observation satisfying 
  #' `colnames(pseudobulks) == rownames(meta)` and contain untransformed counts.
  #' @param meta A `data.frame`. Table containing the per observation covariates 
  #' included in the model. If input is a matrix, `meta` is required. If input 
  #' is a list, `meta` is expected for each entry.
  #' @param model A formula. Constructed with `as.formula()` or `~` and 
  #' containing terms in `colnames(meta)` or `colnames(pseudobulks$meta)`.
  #' @param contrast Numeric or character. Position or name of the contrast in 
  #' `colnames(design.matrix(model, data = meta))`.
  #' @param verbose Logical. Indicates the level of information displayed.
  #' @param return_interm Logical. If TRUE, returns intermediate files.
  #'
  #' Inspired by [Libra pseudobulk differential expression]
  #' (https://github.com/neurorestore/Libra/blob/main/R/pseudobulk_de.R)

  if (verbose) {
    message("Running differential expression...")
  }
  if (is.list(input)) {
    cell_types_expr <- purrr::map(input, "expr")
    cell_types_meta <- purrr::map(input, "meta")
  } else if (is.matrix(input) && is.data.frame(meta)) {
    cell_type_label <- if (is.null(cell_type)) "" else cell_type
    cell_types_expr <- setNames(list(input), cell_type_label)
    cell_types_meta <- setNames(list(meta), cell_type_label)
  } else {
    stop("\"pseudobulks\" must be a list of counts per cell type OR a pair of count matrix and metadata data frame.")
  }

  des <- purrr::pmap(
    list(cell_types_expr, cell_types_meta, names(cell_types_expr)),
    function(expr, meta, cell_type) {
      if (verbose) {
        message(cell_type)
      }
      # Check inputs
      if (!identical(sort(colnames(expr)), sort(rownames(meta)))) {
        stop("Observation names between expression and metadata do not match.")
      }
      covs <- all.vars(model)
      if (!all(covs %in% colnames(meta))) {
        stop(paste0("Model terms absent from meta: ",
                    colnames(meta)[!covs %in% colnames(meta)]))
      }
      if (!all(expr %% 1 == 0)) {
        stop("Counts are not integer.")
      }
      # Ensure design matrix is full rank
      design_full <- model.matrix(model, data = meta)
      stopifnot(identical(rownames(design_full), colnames(expr)))
      design_full <- check_nonestimable(design_full)

      # Check categorical or continuous contrast
      if (!contrast %in% colnames(design_full)) {
        stop(paste0(contrast,
                    " contrast was not present in the full design with terms: ",
                    paste(colnames(design_full), collapse = ", "), "\n"))
      }
      is.categorical <- length(unique(design_full[, contrast])) == 2
      if (!is.categorical & de_type == "voombygroup") {
        message("voombygroup does not support continuous contrasts. Running voom instead.")
        de_type <- "voom"
      }
      # Define reduced design for LRT de_type in DESeq2
      if (is.character(contrast)) {
        design_reduced <- design_full[, colnames(design_full) != contrast]
      } else if (is.numeric(contrast)) {
        design_reduced <- design_full[, -contrast]
      }

      if (verbose) print(data.frame(design_full))

      # Run DE
      de <- switch(
        de_method,
        edgeR = {
          tryCatch(
            {
              y <- edgeR::DGEList(
                expr = expr,
                group = design_full[, contrast]
              ) %>%
                edgeR::calcNormFactors(method = "TMM") %>%
                edgeR::estimateDisp(design_full)
              test <- switch(de_type,
                QLF = {
                  fit <- edgeR::glmQLFit(y, design = design_full)
                  test <- edgeR::glmQLFTest(fit, coef = contrast)
                },
                LRT = {
                  fit <- edgeR::glmFit(y, design = design_full)
                  test <- edgeR::glmLRT(fit, coef = contrast)
                }
              )
              res <- edgeR::topTags(test, n = Inf) %>%
                as.data.frame() %>%
                tibble::rownames_to_column("gene") %>%
                dplyr::mutate(Bonferroni = p.adjust(PValue, "bonferroni"),
                              contrast = contrast) %>%
                dplyr::rename(dplyr::any_of(c(
                  FDR = "FDR", logFC = "logFC", AveEx = "logCPM",
                  P = "PValue", stat = ifelse(de_type == "LRT", "LR", "F")
                ))) %>%
                dplyr::arrange(p.value)
              list(res = res,
                   interm = list(y = y, test = test, fit = fit))
            },
            error = function(e) {
              message(e)
              list()
            }
          )
        },
        DESeq2 = {
          tryCatch(
            {
              dds <- suppressMessages(DESeq2::DESeqDataSetFromMatrix(
                countData = expr, colData = meta, design = design_full
              ))
              dds <- switch(de_type,
                Wald = {
                  dds <- try(DESeq2::DESeq(dds,
                    test = "Wald",
                    quiet = !verbose
                  ))
                },
                LRT = {
                  dds <- try(DESeq2::DESeq(dds,
                    test = "LRT", quiet = !verbose,
                    reduced = design_reduced
                  ))
                }
              )
              res <- DESeq2::results(dds,
                name = contrast, cooksCutoff = F,
                format = "DataFrame"
              ) %>%
                as.data.frame() %>%
                tibble::rownames_to_column("gene") %>%
                dplyr::mutate(
                  Bonferroni = p.adjust(pvalue, "bonferroni"),
                  contrast = contrast
                ) %>%
                dplyr::rename(dplyr::any_of(c(
                  FDR = "padj", logFC = "log2FoldChange", SE = "lfcSE",
                  AveEx = "baseMean", P = "pvalue"
                ))) %>%
                dplyr::arrange(P) %>%
                tidyr::drop_na()
              list(res = res, interm = dds)
            },
            error = function(e) {
              message(e)
              list()
            }
          )
        },
        limma = {
          tryCatch(
            {
              # Scale library size
              norm.factors <- edgeR::calcNormFactors(expr)
              lib.size <- norm.factors * colSums(expr)
              # voom or trend
              switch(de_type,
                trend = {
                  trend_bool <- T
                  dge <- edgeR::DGEList(expr, lib.size = lib.size,
                                        norm.factors = norm.factors,
                                        group = design_full[, contrast])
                  v <- methods::new("EList")
                  v$E <- edgeR::cpm(dge, log = TRUE)
                  v
                },
                voom = {
                  trend_bool <- F
                  v <- limma::voom(expr, design_full, save.plot = TRUE,
                                   lib.size = lib.size)
                  voom_plot <- cbind.fill(
                    voom_plot_x = v$voom.xy$x,
                    voom_plot_y = v$voom.xy$y,
                    voom_line_x = v$voom.line$x,
                    voom_line_y = v$voom.line$y
                  ) %>%
                    tibble::rownames_to_column("gene")
                },
                voombygroup = {
                  trend_bool <- F
                  v <- voomByGroup(expr, design_full, save.plot = TRUE,
                                   plot = "combine", print = TRUE,
                                   group = design_full[, contrast],
                                   lib.size = lib.size)
                  voom_plot <- cbind.fill(
                    voom_plot_1_x = v$voom.xy[[1]]$x,
                    voom_plot_1_y = v$voom.xy[[1]]$y,
                    voom_plot_2_x = v$voom.xy[[2]]$x,
                    voom_plot_2_y = v$voom.xy[[2]]$y,
                    voom_line_1_x = v$voom.line[[1]]$x,
                    voom_line_1_y = v$voom.line[[1]]$y,
                    voom_line_2_x = v$voom.line[[2]]$x,
                    voom_line_2_y = v$voom.line[[2]]$y
                  ) %>%
                    tibble::rownames_to_column("gene")
                }
              )
              # get fit
              fit <- limma::lmFit(v, design_full) %>%
                limma::eBayes(trend = trend_bool, robust = trend_bool)
              # calculate standard errors for the coefficients
              SE <- sqrt(fit$s2.post) * fit$stdev.unscaled
              # format the results
              res <- fit %>%
                limma::topTable(
                  number = Inf, coef = contrast, confint = TRUE) %>%
                {
                  if (nrow(.) == nrow(SE)) {
                    dplyr::bind_cols(., SE = SE[, contrast])
                  } else {
                    .
                  }
                } %>%
                tibble::rownames_to_column("gene") %>%
                dplyr::left_join(voom_plot, by = "gene") %>%
                {
                  if (is.categorical) {
                    dplyr::mutate(
                      .,
                      n0 = sum(design_full[, contrast, drop = TRUE] == 0),
                      n1 = sum(design_full[, contrast, drop = TRUE] == 1)
                    )
                  } else {
                    .
                  }
                } %>%
                dplyr::mutate(
                  B = p.adjust(P.Value, "bonferroni"),
                  contrast = contrast
                ) %>%
                dplyr::rename(dplyr::any_of(c(
                  FDR = "adj.P.Val", logFC = "logFC", LCI= "CI.L", 
                  Bonferroni = "B", UCI= "CI.R", AveExpr = "AveExpr", 
                  P = "P.Value",  stat = "t" ))) %>%
                dplyr::arrange(P)
              list(res = res, interm = list(v = v, fit = fit))
            },
            error = function(e) {
              message(e)
              list()
            }
          )
        }
      )
      de[["design_full"]] <- design_full
      de[["design_reduced"]] <- design_reduced

      return(de)
    }
  )

  # Return
  if (return_interm) {
    des
  } else {
    purrr::map_dfr(des, "res", .id = "cell_type")
  }
}

# https://github.com/tluquez/utils/blob/cbb0392ae223da31014fa17bdb6975247557731a/utils.R#L1712
#' Diagnose RUVSeq
#'
#' @description
#' Returns diagnostic plots and metrics to evaluate the impact of addign RUVseq
#' factors to a differential expression analysis.on the number of DEGs.
#'
#' @param input list or matrix. List of cell types with list of expr and meta
#' generated by [Libra::to_pseudobulk()].`rownames(pseudobulks$meta)` must 
#' match `colnames(pseudobulks$expr)`. It can also be a count matrix of 
#' gene x observation satisfying `colnames(pseudobulks) == rownames(meta)` 
#' and raw counts.
#' @param meta data.frame. Table containing the per observation covariates
#' included in model. If input is a matrix, meta is required. 
#' Is input is a list, meta is expected per each entry.
#' @param model formula. Constructed with [as.formula()] or `~` and 
#' containing terms in `colnames(meta)` or `colnames(pseudobulks$meta)`.
#' @param contrast numeric or character. Position or name of the contrast 
#' in `colnames(design.matrix(model, data = meta))`.
#' @param ruv_type character. One of `c("RUVr", "RUVg", "RUVs")`.
#' @param cIdx, @param scIdx, @param resids As in [RUVSeq::RUVs()].
#' @param min_k, @param max_k numeric. Number of factors to use for [RUVSeq]. 
#' If min_k != max_k it will run all the factors. If min_k == max_k it will 
#' only run one iteration
#' @param de_method, de_type As in [de_pseudobulk].
#' @param plot_pcs logical. Whether to plot PCs for factor 1 and 2 for all 
#' covariates in model.
#' @param plot_pcs_dir character. Path to file to write PC analysis 
#' PDFs if `plot_pcs = T`.
#' @param plot_voom_dir
#' @param verbose logical. Level fo information displayed. 0 to silence, 
#' 1 to display progress information and cell type names, 2 to display 
#' per k iteration ifnormation and 3 to print matrix of DEG by k.
#'
#' @returns
#' List per cell type with a list containing de_method x de_type data frame 
#' of DEGs (`degs`), a data frame of all genes (`de`), a data frame of 
#' k vs number of DEGs (`k_vs_ndegs`), a data frame of k vs total explained 
#' variance by covariates (`varparts`), nominal p-value distributions
#' (`pval_quantiles`), pearson correlation of the differential expression 
#' statistic without and with RUVSeq factors in the model across all genes 
#' (`stat_cors`) and the k that maximizes the variance explained for the term 
#' of the contrast (`best_k`).

diagnose_ruv <- function(input, meta,  model, contrast, cell_type,
                         ruv_type = "RUVr", cIdx = NULL, scIdx = NULL,
                         resids = NULL, min_k = 5, max_k = 5,
                         de_method = "limma", de_type = "voombygroup",
                         plot_pcs = FALSE, plot_pcs_dir = NULL, 
                         plot_voom_dir = NULL, 
                         verbose = 1) {

  library(ggplot2)

  # Intialize output
  if (is.null(plot_pcs_dir)) {
    plot_pcs_dir <- paste0(getwd(), "/diagnose_", ruv_type)
  }
  base_dir <- dirname(plot_pcs_dir)

  if (verbose >= 1) cat("Baseline differential expression")
  png(paste0(plot_voom_dir, "/", cell_type, "_k0.png"), 
      height = 5, width = 6, units = "in", res = 300)
  des <- de_pseudobulk(input = input, meta = meta, model = model,
                       cell_type = cell_type, contrast = contrast,
                       de_method = de_method, de_type = de_type,
                       verbose = F, return_interm = T)
  dev.off()
  # Loop over each cell type
  furrr::future_imap(des, function(res, cell_type) {
    if (verbose >= 1) cat(paste0("Diagnosis RUVSeq for cell type: ", cell_type,
                                 "\n"))

    # Get normalized and log transformed counts
    expr <- switch(
      de_method,
      edgeR = edgeR::cpm(res$interm$y, log = TRUE),
      DESeq2 = DESeq2::rlog(DESeq2::counts(res$interm$dds, normalized = TRUE)),
      limma = res$interm$v$E
    )

    # Get residuals if not supplied and ruv_type == "RUVr"
    if (is.null(resids) & ruv_type == "RUVr") {
      resids <- switch(
        de_method,
        edgeR = residuals(res$interm$fit),
        DESeq2 = SummarizedExperiment::assays(res$interm$dds)[["mu"]],
        limma = residuals(res$interm$fit, res$interm$v)
      )
    }

    # Store design matrix in case verbose >= 3 for gene count plots
    res_design_full <- res$design_full

    # Get the same cell_type name as de_pseudobulk
    res <- dplyr::bind_rows(res$res, .id = "cell_type")
    qs <- quantile(res$P)

    # Get variance partition
    varpart <- variancePartition::fitExtractVarPartModel(expr, model, meta)
    colnames(varpart) <- make.names(colnames(varpart), unique = TRUE)
    varpartmeds <- Rfast::colMedians(varpart)

    # Store k = 0 results
    k0 <- list(
      degs = res[res$FDR < .05, ],
      de = res,
      varparts = c(varpartmeds,
                   total = sum(varpartmeds[names(varpartmeds) != "Residuals"])),
      k_vs_ndegs = nrow(res[res$FDR < .05, ]),
      pval_quantiles = data.frame(
        q0 = qs[1], q25 = qs[2], q50 = qs[3],
        q75 = qs[4], q100 = qs[5], row.names = NULL
      )
    )

    # Run iterations
    if (verbose >= 1) cat(paste0("RUVseq with max_k: ", max_k, "\n"))
    ks_res <- furrr::future_map(min_k:max_k, function(k) {
      if (verbose >= 3) cat(paste0("\n    k: ", k, "\n"))

      # Define RUV type
      ks <- switch(
        ruv_type,
        RUVg = RUVSeq::RUVg(x = expr, cIdx = cIdx, k = k, round = F, isLog = T),
        RUVs = RUVSeq::RUVs(x = expr, cIdx = rownames(expr), scIdx = scIdx,
                            k = k, round = F, isLog = T),
        RUVr = RUVSeq::RUVr(x = expr, cIdx = rownames(expr), k = k, round = F,
                            residuals = resids, isLog = T)
      )

      if (plot_pcs) {
        if (verbose >= 3) cat("    Plotting PCs\n")
        pc <- prcomp(t(ks$normalizedCounts), center = T, scale. = T)
        summary(pc)$importance[, 1:3]
        sort(pc$rotation[, 1], decreasing = T)[1:20]
        suppressMessages(
          purrr::imap(colnames(meta), ~ {
            ggplot(cbind(pc$x[, 1:2], meta),
                   aes(PC1, PC2, color = .data[[.x]])) +
              geom_point() +
              labs(title = paste("Number of ks: ", k)) +
              theme_classic()
            ggsave(paste0(base_dir, "/temp_", .y, "_", k, ".pdf"))
          })
        )
        l <- list.files(path = base_dir, pattern = "temp_", full.names = T)
        qpdf::pdf_combine(l, paste0(plot_pcs_dir, "_pcs", k, "_2.pdf"))
        qpdf::pdf_compress(
          paste0(plot_pcs_dir, "_pcs", k, "_2.pdf"),
          paste0(plot_pcs_dir, "_pcs", k, ".pdf")
        )
        invisible(file.remove(c(l, paste0(plot_pcs_dir, "_pcs", k, "_2.pdf"))))
        rm(l)
      }

      # Include RUV factors in DE
      model2 <- as.formula(
        paste(
          "~",
          paste(
            as.character(model)[2],
            paste(paste0("W_", 1:k), collapse = " + "),
            sep = " + "
          )
        )
      )
      if (is.list(input)) {
        counts <- input[[cell_type]][["expr"]]
        meta2 <- cbind(input[[cell_type]][["meta"]], ks$W)
      } else if (is.matrix(input) & is.data.frame(meta)) {
        counts <- input
        meta2 <- cbind(meta, ks$W)
      }
      png(paste0(plot_voom_dir, "/", cell_type, "_k", k, ".png"), 
          height = 5, width = 6, units = "in", res = 300)
      res2 <- de_pseudobulk(input = counts, meta = meta2, model = model2,
                            cell_type = cell_type, contrast = contrast, 
                            de_method = de_method, de_type = de_type,
                            verbose = F, return_interm = F)
      dev.off()
      # Get DE diagnostics
      qs <- quantile(res2$P)
      stat_cor <- cor(abs(res$stat), abs(res2$stat), method = "spearman")

      # Variance partition
      if (verbose >= 3) cat("    Variance partition\n")
      varpart <- variancePartition::fitExtractVarPartModel(expr, model2, meta2)
      colnames(varpart) <- make.names(colnames(varpart), unique = T)
      varpart_covs <- colnames(varpart)
      if (verbose >= 3) {
        for (cov in varpart_covs) {
          assign(cov, varpart[, cov])
        }
        eval(parse(text = paste0("txtplot::txtboxplot(",
                                 paste(varpart_covs, collapse = ", "), ")")))
        rm(list = varpart_covs)
      }
      varpartmeds <- Rfast::colMedians(varpart)

      # Display information
      if (verbose >= 3) {
        cat(paste0("    DEGs (n): ", nrow(res2[res2$FDR < .05, ]), "\n"))
        cat(paste0("    Median p-value: ", median(res2$P), "\n"))
        cat("    p-value distribution: \n")
        print(txtplot::txtdensity(res2$P))
        cat(paste0(
          "    Correlation between the uncorrected and corrected t: ",
          stat_cor, "\n"
        ))
        print(txtplot::txtplot(abs(res$stat), abs(res2$stat)))
      }

      # Return
      list(
        degs = res2[res2$FDR < .05, ],
        de = res2,
        varparts = c(varpartmeds, total = sum(varpartmeds[names(varpartmeds)
                                                          != "Residuals"])),
        k_vs_ndegs = nrow(res2[res2$FDR < .05, ]),
        pval_quantiles = data.frame(
          q0 = qs[1], q25 = qs[2], q50 = qs[3],
          q75 = qs[4], q100 = qs[5], row.names = NULL
        ),
        stat_cors = stat_cor,
        k_w = ks$W 
      )
    },
    .options = furrr::furrr_options(seed = T)) %>%
      setNames(min_k:max_k)
    ks_res <- c(list("0" = k0), ks_res)

    # Convert output lists into dataframes
    degs <- purrr::map_dfr(ks_res, "degs", .id = "k") %>%
      dplyr::mutate(k = as.integer(k))
    de <- purrr::map_dfr(ks_res, "de", .id = "k") %>%
      dplyr::mutate(k = as.integer(k))
    varparts <- purrr::map_dfr(ks_res, "varparts", .id = "k") %>%
      dplyr::mutate(k = as.integer(k))
    k_vs_ndegs <- purrr::map_dfr(ks_res, ~{
      tibble::enframe(unlist(.x[["k_vs_ndegs"]]), name = "k", value = "n_degs")
    }, .id = "k") %>%
      dplyr::mutate(k = as.integer(k))
    pval_quantiles <- purrr::map_dfr(ks_res, "pval_quantiles", .id = "k") %>%
      dplyr::mutate(k = as.integer(k))
    stat_cors <- purrr::map_dfr(ks_res, ~{
      tibble::enframe(unlist(.x$stat_cors), name = "k", value = "cor")
    }, .id = "k") %>%
      dplyr::mutate(k = as.integer(k))

    # Get k that maximizes the variance explained for the term of interest
    contrast_term <- all.vars(model)[grep(paste(all.vars(model),
                                                collapse = "|"), contrast)]
    best_k <- varparts$k[which.max(varparts[[contrast_term]])]

    if (plot_pcs) {
      # Combine PC PDFs into a single pdf
      l <- list.files(
        path = base_dir, pattern = paste0(basename(plot_pcs_dir), "_pcs"),
        full.names = T
      )
      qpdf::pdf_combine(l, paste0(plot_pcs_dir, "_pcs_2.pdf"))
      qpdf::pdf_compress(
        paste0(plot_pcs_dir, "_pcs_2.pdf"),
        paste0(plot_pcs_dir, "_pcs.pdf")
      )
      invisible(file.remove(c(l, paste0(plot_pcs_dir, "_pcs_2.pdf"))))
      rm(l)
    }

    # Print cross k information
    if (verbose >= 1) {
      cat("RUV k vs n_degs\n")
      print(txtplot::txtplot(k_vs_ndegs$k, k_vs_ndegs$n_degs))

      cat("Total variance explained\n")
      print(txtplot::txtplot(varparts$k, varparts$total))
    }

    if (verbose >= 2) {
      cat("Number of ks supporting a DEG\n")
      k_v_deg <- table(degs$gene, degs$k)
      cbind(k_v_deg, total = rowSums(k_v_deg)) %>%
        as.data.frame() %>%
        dplyr::arrange(total) %>%
        knitr::kable() %>%
        message()
    }

    if (verbose >= 3 & nrow(degs) != 0) {
      is.categorical <- length(unique(res_design_full[, contrast])) == 2
      n_degs <- length(unique(degs$gene))
      max_n_degs <- ifelse(n_degs > 10, 10, n_degs)
      if (is.categorical) {
        # Get sample names per category
        expr_0 <- rownames(res_design_full)[res_design_full[, contrast] == 0]
        expr_1 <- rownames(res_design_full)[res_design_full[, contrast] == 1]

        # Plot each gene
        cat(paste0("Top ", max_n_degs, " DEGs for contrast ", contrast, " with ",
                length(expr_0), " controls and ", length(expr_1), " cases.\n"))
        for (i in 1:max_n_degs) {
          gene <- degs %>%
            dplyr::count(gene) %>%
            dplyr::arrange(dplyr::desc(n)) %>%
            magrittr::extract(i, "gene")
          print(gene)
          print(txtplot::txtboxplot(
            expr[rownames(expr) == gene, expr_0],
            expr[rownames(expr) == gene, expr_1]
          ))
        }
      } else {
        for (i in 1:max_n_degs) {
          gene <- degs %>%
            dplyr::count(gene) %>%
            dplyr::arrange(dplyr::desc(n)) %>%
            magrittr::extract(i, "gene")
          print(gene)
          print(txtplot::txtboxplot(expr[rownames(expr) == gene, ]))
        }
      }
    }

    # Return
    list(
      degs = degs, de = de, k_vs_ndegs = k_vs_ndegs, varparts = varparts,
      pval_quantiles = pval_quantiles, stat_cors = stat_cors, best_k = best_k,
      ks_w = lapply(ks_res, function(x) x$k_w)
    )
  }, .options = furrr::furrr_options(seed = T))
}
