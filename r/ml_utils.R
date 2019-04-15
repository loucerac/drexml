## UTILS
##

get_preds_rfsrc <- function(res, index=NULL) {
    preds <- as.data.frame(lapply(res$regrOutput, function(x) {x$predicted}))
    rownames(preds) <- rownames(res$xvar)

    if (!(is.null(index))) {
        rownames(preds) <- index
    }

    return(preds)
}


r2_score <- function(y_true, y_pred, multioutput="raw") {
    numerator <- colSums((y_true - y_pred) ^ 2)
    denominator <- colSums(sweep(y_true, 2, colMeans(y_true), `-`) ^ 2)

    nonzero_denominator <- denominator != 0
    nonzero_numerator <- numerator != 0
    valid_score <- nonzero_denominator & nonzero_numerator
    output_scores = rep(1.0, ncol(y_true))
    output_scores[valid_score] <- 1 - (numerator[valid_score] / denominator[valid_score])
    # arbitrary set to zero to avoid -inf scores, having a constant
    # y_true is not interesting for scoring a regression anyway
    output_scores[nonzero_numerator & !nonzero_denominator] <- 0.0

    if (multioutput == "raw") {
        val <- output_scores
    } else {
        val <- mean(output_scores)
    }

    return(val)
}
