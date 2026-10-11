# GLM-Based Claims Reserving with Bootstrap
# Reads a claims triangle from CSV and policy data from SQLite,
# fits an over-dispersed Poisson GLM, and performs bootstrap
# to estimate reserve distributions with confidence intervals.

library(DBI)
library(RSQLite)
library(dplyr)

# ---- 1. Load claims triangle from CSV ----
triangle_raw <- read.csv("claims_triangle.csv", row.names = 1)
triangle_inc <- as.matrix(triangle_raw)

cat("Claims triangle loaded:", nrow(triangle_inc), "x", ncol(triangle_inc), "\n")

# ---- 2. Load supplementary policy data from SQLite ----
load_policy_data <- function(db_path) {
  con <- dbConnect(SQLite(), db_path)

  tryCatch({
    policies <- dbGetQuery(con, "SELECT * FROM policies")
    premiums <- dbGetQuery(con, "SELECT * FROM premiums")
    claims <- dbGetQuery(con, "SELECT * FROM claims_detail")

    # Aggregate earned premium by origin year
    policy_premiums <- merge(policies, premiums, by = "policy_id")
    premium_by_year <- policy_premiums %>%
      group_by(origin_year) %>%
      summarise(
        total_earned_premium = sum(earned_premium),
        total_written_premium = sum(written_premium),
        n_policies = n(),
        .groups = "drop"
      )

    cat("Policy data loaded:", nrow(policies), "policies,", nrow(claims), "claims\n")

    return(list(
      premium_by_year = as.data.frame(premium_by_year),
      policies = policies,
      claims = claims
    ))
  }, finally = {
    dbDisconnect(con)
  })
}

policy_data <- load_policy_data("policies.db")
premium_by_year <- policy_data$premium_by_year

cat("\nPremium summary by year:\n")
print(head(premium_by_year, 5))

# ---- 3. Prepare triangle data for GLM ----
prepare_glm_data <- function(triangle) {
  n_rows <- nrow(triangle)
  n_cols <- ncol(triangle)

  records <- data.frame(
    origin = integer(),
    dev = integer(),
    incremental = numeric()
  )

  for (i in 1:n_rows) {
    for (j in 1:n_cols) {
      if (!is.na(triangle[i, j])) {
        records <- rbind(records, data.frame(
          origin = i,
          dev = j,
          incremental = triangle[i, j]
        ))
      }
    }
  }

  # Convert to factors for GLM
  records$origin_f <- as.factor(records$origin)
  records$dev_f <- as.factor(records$dev)

  return(records)
}

glm_data <- prepare_glm_data(triangle_inc)
cat("\nGLM data prepared:", nrow(glm_data), "observations\n")

# ---- 4. Fit Over-Dispersed Poisson GLM ----
fit_odp_glm <- function(data) {
  # Fit quasi-Poisson GLM (over-dispersed Poisson)
  model <- glm(
    incremental ~ origin_f + dev_f,
    data = data,
    family = quasipoisson(link = "log")
  )

  # Extract dispersion parameter (phi)
  phi <- summary(model)$dispersion

  cat("GLM fitted successfully\n")
  cat("  Dispersion parameter (phi):", round(phi, 4), "\n")
  cat("  Residual deviance:", round(model$deviance, 2), "\n")
  cat("  Degrees of freedom:", model$df.residual, "\n")

  return(list(model = model, phi = phi))
}

glm_result <- fit_odp_glm(glm_data)
model <- glm_result$model
phi <- glm_result$phi

# ---- 5. Predict missing values (lower triangle) ----
predict_reserves <- function(model, n_rows, n_cols) {
  # Create prediction data for the lower triangle
  pred_data <- data.frame(
    origin = integer(),
    dev = integer()
  )

  for (i in 2:n_rows) {
    start_dev <- n_cols - i + 2
    if (start_dev <= n_cols) {
      for (j in start_dev:n_cols) {
        pred_data <- rbind(pred_data, data.frame(origin = i, dev = j))
      }
    }
  }

  pred_data$origin_f <- as.factor(pred_data$origin)
  pred_data$dev_f <- as.factor(pred_data$dev)

  # Predict on log scale, then exponentiate
  pred_data$predicted <- predict(model, newdata = pred_data, type = "response")

  return(pred_data)
}

n_rows <- nrow(triangle_inc)
n_cols <- ncol(triangle_inc)
predictions <- predict_reserves(model, n_rows, n_cols)

# Calculate reserves by origin year
reserves_by_year <- predictions %>%
  group_by(origin) %>%
  summarise(reserve = sum(predicted), .groups = "drop")

total_reserve <- sum(reserves_by_year$reserve)
cat("\nReserves by origin year:\n")
print(reserves_by_year)
cat("\nTotal reserve (point estimate):", round(total_reserve, 2), "\n")

# ---- 5b. Deterministic outputs (for cross-language comparison) ----
cat("\n==== DETERMINISTIC OUTPUTS ====\n")
cat("Dispersion parameter (phi):", sprintf("%.10f", phi), "\n")
cat("Total reserve (point estimate):", sprintf("%.6f", total_reserve), "\n")
cat("\nReserves by origin year:\n")
for (i in 1:nrow(reserves_by_year)) {
  cat(sprintf("  Origin %d: %.6f\n", reserves_by_year$origin[i], reserves_by_year$reserve[i]))
}
cat("==== END DETERMINISTIC OUTPUTS ====\n")

# ---- 6. Bootstrap for Reserve Distribution ----
bootstrap_reserves <- function(model, glm_data, triangle, n_boot = 100, phi) {
  n_rows <- nrow(triangle)
  n_cols <- ncol(triangle)

  # Get Pearson residuals
  fitted_vals <- fitted(model)
  raw_residuals <- (glm_data$incremental - fitted_vals) / sqrt(fitted_vals)

  # Adjust residuals for degrees of freedom
  n_params <- length(coef(model))
  n_obs <- nrow(glm_data)
  adj_factor <- sqrt(n_obs / (n_obs - n_params))
  adj_residuals <- raw_residuals * adj_factor

  # Remove any NA or Inf residuals
  valid_residuals <- adj_residuals[is.finite(adj_residuals)]

  set.seed(42)
  boot_reserves <- numeric(n_boot)

  for (b in 1:n_boot) {
    # Resample residuals
    resampled_residuals <- sample(valid_residuals, length(fitted_vals), replace = TRUE)

    # Create pseudo-data
    pseudo_incremental <- fitted_vals + resampled_residuals * sqrt(fitted_vals)
    pseudo_incremental <- pmax(pseudo_incremental, 1)  # Ensure positive

    pseudo_data <- glm_data
    pseudo_data$incremental <- pseudo_incremental

    # Refit model
    tryCatch({
      pseudo_model <- glm(
        incremental ~ origin_f + dev_f,
        data = pseudo_data,
        family = quasipoisson(link = "log")
      )

      # Predict reserves
      pred <- predict_reserves(pseudo_model, n_rows, n_cols)

      # Add process variance (gamma random)
      pred$simulated <- rgamma(
        nrow(pred),
        shape = pred$predicted / phi,
        scale = phi
      )

      boot_reserves[b] <- sum(pred$simulated)
    }, error = function(e) {
      boot_reserves[b] <<- NA
    })
  }

  return(boot_reserves[!is.na(boot_reserves)])
}

cat("\nRunning bootstrap (1000 simulations)...\n")
boot_results <- bootstrap_reserves(model, glm_data, triangle_inc, n_boot = 1000, phi = phi)

cat("Bootstrap complete:", length(boot_results), "successful simulations\n")

cat("\n==== STOCHASTIC OUTPUTS ====\n")

# ---- 7. Summary statistics ----
reserve_summary <- data.frame(
  statistic = c("Mean", "Std Dev", "CV", "P25", "P50 (Median)", "P75", "P95", "P99"),
  value = round(c(
    mean(boot_results),
    sd(boot_results),
    sd(boot_results) / mean(boot_results),
    quantile(boot_results, 0.25),
    quantile(boot_results, 0.50),
    quantile(boot_results, 0.75),
    quantile(boot_results, 0.95),
    quantile(boot_results, 0.99)
  ), 2)
)

cat("\nReserve Distribution Summary:\n")
print(reserve_summary)

# ---- 8. Merge with premium data for reserve-to-premium ratios ----
origin_years <- as.integer(rownames(triangle_raw))
reserves_with_premium <- merge(
  data.frame(origin_year = origin_years[reserves_by_year$origin],
             reserve = reserves_by_year$reserve),
  premium_by_year,
  by = "origin_year",
  all.x = TRUE
)

reserves_with_premium$reserve_to_premium <- round(
  reserves_with_premium$reserve / reserves_with_premium$total_earned_premium, 4
)

cat("\nReserves with Premium Ratios:\n")
print(reserves_with_premium[, c("origin_year", "reserve", "total_earned_premium", "reserve_to_premium")])

# ---- 9. Save results ----
write.csv(reserve_summary, "reserve_summary.csv", row.names = FALSE)
cat("\nResults saved to reserve_summary.csv\n")
