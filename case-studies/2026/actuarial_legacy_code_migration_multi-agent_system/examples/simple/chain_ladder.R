options(stringsAsFactors = FALSE)

# ---------------------------
# User options
# ---------------------------
input_file  <- "triangle.csv"
tail_factor <- 1.00   # 1.00 = no tail beyond the last development period

# ---------------------------
# Read incremental triangle
# ---------------------------
triangle_data <- read.csv(input_file, row.names = 1, check.names = FALSE)
triangle_inc  <- data.matrix(triangle_data)
storage.mode(triangle_inc) <- "numeric"

origin_years <- rownames(triangle_inc)
dev_periods  <- colnames(triangle_inc)

n_origin <- nrow(triangle_inc)
n_dev    <- ncol(triangle_inc)

if (n_origin == 0 || n_dev < 2) {
  stop("Triangle must have at least 1 origin period and 2 development periods.")
}

if (any(origin_years == "")) {
  stop("Origin period names are missing.")
}

if (any(rowSums(!is.na(triangle_inc)) == 0)) {
  stop("At least one origin period has no observed data.")
}

# ---------------------------
# Validate triangle structure
# ---------------------------
row_is_contiguous <- function(x) {
  obs <- which(!is.na(x))
  if (length(obs) == 0) return(FALSE)
  all(obs == seq_len(max(obs)))
}

bad_rows <- which(!apply(triangle_inc, 1, row_is_contiguous))
if (length(bad_rows) > 0) {
  stop(
    paste0(
      "Triangle has internal missing values or does not start at development period 1 for row(s): ",
      paste(origin_years[bad_rows], collapse = ", "),
      ". Clean the input triangle before reserving."
    )
  )
}

# Optional warning on negative increments
if (any(triangle_inc[!is.na(triangle_inc)] < 0)) {
  warning(
    "Negative incremental values detected. ",
    "Simple chain-ladder on cumulative data may be unstable; review recoveries/re-openings."
  )
}

# ---------------------------
# Compute cumulative triangle
# ---------------------------
triangle_cum <- matrix(
  NA_real_,
  nrow = n_origin,
  ncol = n_dev,
  dimnames = list(origin_years, dev_periods)
)

for (i in seq_len(n_origin)) {
  obs <- which(!is.na(triangle_inc[i, ]))
  triangle_cum[i, obs] <- cumsum(triangle_inc[i, obs])
}

# Warn if cumulative triangle is not non-decreasing
for (i in seq_len(n_origin)) {
  obs <- which(!is.na(triangle_cum[i, ]))
  if (length(obs) > 1 && any(diff(triangle_cum[i, obs]) < 0)) {
    warning(
      sprintf(
        "Cumulative claims decrease for origin period %s. Review input data.",
        origin_years[i]
      )
    )
  }
}

cat("Cumulative Triangle:\n")
print(triangle_cum)

# ---------------------------
# Calculate age-to-age factors
# ---------------------------
# Simple volume-weighted average, using only rows where both ages are observed.
dev_factors <- rep(NA_real_, n_dev - 1)
n_pairs     <- integer(n_dev - 1)

for (j in seq_len(n_dev - 1)) {
  valid_rows <- !is.na(triangle_cum[, j]) & !is.na(triangle_cum[, j + 1])
  n_pairs[j] <- sum(valid_rows)
  
  if (n_pairs[j] == 0) {
    stop(sprintf(
      "No observed pairs available to calculate development factor for %s -> %s.",
      dev_periods[j], dev_periods[j + 1]
    ))
  }
  
  denom <- sum(triangle_cum[valid_rows, j])
  
  if (denom <= 0) {
    stop(sprintf(
      "Non-positive denominator when calculating development factor for %s -> %s.",
      dev_periods[j], dev_periods[j + 1]
    ))
  }
  
  dev_factors[j] <- sum(triangle_cum[valid_rows, j + 1]) / denom
  
  if (dev_factors[j] < 1) {
    warning(sprintf(
      "Selected development factor for %s -> %s is %.4f (< 1). Review whether simple chain-ladder is appropriate.",
      dev_periods[j], dev_periods[j + 1], dev_factors[j]
    ))
  }
}

dev_factor_table <- data.frame(
  from_dev  = dev_periods[1:(n_dev - 1)],
  to_dev    = dev_periods[2:n_dev],
  n_pairs   = n_pairs,
  factor    = round(dev_factors, 6),
  row.names = NULL
)

cat("\nDevelopment Factors (volume-weighted):\n")
print(dev_factor_table)

# ---------------------------
# Calculate cumulative development factors (CDF) to ultimate
# ---------------------------
cdf_to_ultimate <- rep(NA_real_, n_dev)
cdf_to_ultimate[n_dev] <- tail_factor

if (n_dev > 1) {
  for (j in (n_dev - 1):1) {
    cdf_to_ultimate[j] <- dev_factors[j] * cdf_to_ultimate[j + 1]
  }
}

cdf_table <- data.frame(
  dev_period      = dev_periods,
  cdf_to_ultimate = round(cdf_to_ultimate, 6),
  row.names       = NULL
)

cat("\nCDF to Ultimate:\n")
print(cdf_table)

# ---------------------------
# Project ultimate claims
# ---------------------------
latest_observed <- rep(NA_real_, n_origin)
latest_dev_idx  <- integer(n_origin)
ultimate_claims <- rep(NA_real_, n_origin)
reserve_amount  <- rep(NA_real_, n_origin)

for (i in seq_len(n_origin)) {
  obs <- which(!is.na(triangle_cum[i, ]))
  latest_dev_idx[i]  <- max(obs)
  latest_observed[i] <- triangle_cum[i, latest_dev_idx[i]]
  ultimate_claims[i] <- latest_observed[i] * cdf_to_ultimate[latest_dev_idx[i]]
  reserve_amount[i]  <- ultimate_claims[i] - latest_observed[i]
}

# ---------------------------
# Results
# ---------------------------
results <- data.frame(
  origin_year            = origin_years,
  latest_dev_period      = dev_periods[latest_dev_idx],
  latest_observed        = round(latest_observed, 2),
  cdf_to_ultimate        = round(cdf_to_ultimate[latest_dev_idx], 6),
  ultimate               = round(ultimate_claims, 2),
  unpaid_claims_reserve  = round(reserve_amount, 2),
  row.names              = NULL
)

cat("\nUltimate Claims by Origin Year:\n")
print(results)

cat(
  "\nTotal Unpaid Claims Reserve:",
  round(sum(results$unpaid_claims_reserve, na.rm = TRUE), 2),
  "\n"
)