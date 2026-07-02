# get ready
rm(list=ls())
set.seed(4228) # for replication

# load packages 
pacman::p_load(tidyverse, emmeans, tidyr, dplyr, knitr, lme4, afex, lmerTest, pbapply)
options(knitr.kable.NA = '') # hide NA with knitr function

datapath <- '../data_pilot/'
DataFiles <- list.files(path = datapath, pattern = "*.csv", full.names = TRUE)

Total_DataL <- data.frame()
sub_useL <- data.frame()


# Loop through each participant's data file
for (k in 1:length(DataFiles)) {
  
  participant_data <- read.csv(DataFiles[k])
  participant_data <- subset(participant_data, select = c("participant","age", "sex", "task", "rt", "stimulus",
                                                          "response", "block","attentioncheck", "trial",
                                                          "curvature","vertices","range"))
  participant_data <- participant_data %>% filter(task == 'test')
  participant_data <- participant_data %>% filter(attentioncheck == '0')
  participant_data <- participant_data %>% filter(response != 'null')
  
  participant_data$stimulus <- gsub('stimuli/experiment_images/abstract_shapes_black/', '', participant_data$stimulus)
  
  participant_data <- participant_data %>%
    mutate(size = case_when(
      str_detect(stimulus, "small") ~ "small",
      str_detect(stimulus, "big") ~ "big",
      TRUE ~ NA_character_
    ))
  
  participant_data <- participant_data %>%
    mutate(image = str_remove(stimulus, "_(big|small)\\.png$"))
  
  Total_DataL <- rbind(Total_DataL, participant_data)
}

# tidy & factor conversions
Total_DataL <- Total_DataL %>%
  mutate(
    participant = as.factor(participant),
    image = as.factor(image),
    size = factor(size, levels = c("small","big")),
    range = as.factor(range),
    vertices = as.factor(vertices),
    curvature = as.factor(curvature),
    rating = as.numeric(response)  # use response column as rating (ensure numeric)
  )

# quick sanity check
cat("Participants:", nlevels(Total_DataL$participant), "\n")
cat("Items (images):", nlevels(Total_DataL$image), "\n")
cat("Rows:", nrow(Total_DataL), "\n")

condition_summary <- Total_DataL %>%
  group_by(size, curvature) %>%
  summarise(mean_value = mean(rating, na.rm = TRUE),
            sd_value = sd(rating, na.rm = TRUE),
            se_value = sd(rating, na.rm = TRUE) / sqrt(n()),
            .groups = "drop")
condition_summary


# ---------------- 1) Fit LMM to pilot to extract variance components & fixed effects ------------

# Fit the pilot model (random intercepts for subj and item)
# Use REML for variance component estimation
m_pilot <- lmer(rating ~ size * curvature + ( 1 | participant) + (1 | image), 
                data = Total_DataL, REML = TRUE)

print(summary(m_pilot))

# Extract estimates for simulation
fixefs <- fixef(m_pilot)
est_size <- ifelse("sizebig" %in% names(fixefs), fixefs["sizebig"], 0)
est_curvature <- ifelse("curvaturesmooth" %in% names(fixefs), fixefs["curvaturesmooth"], 0)

vc <- as.data.frame(VarCorr(m_pilot))
SD_subj <- as.numeric(attr(VarCorr(m_pilot)$participant, "stddev"))
SD_item <- as.numeric(attr(VarCorr(m_pilot)$image, "stddev"))
SD_resid <- sigma(m_pilot)

cat(sprintf("Pilot estimates:\n size effect = %.4f (raw)\n curvature effect = %.4f (raw)\n SD_subj = %.4f, SD_item = %.4f, SD_resid = %.4f\n",
            est_size, est_curvature, SD_subj, SD_item, SD_resid))

# ---------------- 2) Simulation function (factor-based) ----------------
simulate_once <- function(Nsubj,
                          size_effect = est_size,
                          curvature_effect = est_curvature,
                          sd_subj = SD_subj,
                          sd_item = SD_item,
                          sd_resid = SD_resid,
                          sets = 3) {
  
  size_levels <- levels(Total_DataL$size)    
  curvature_levels <- levels(Total_DataL$curvature)
  verts <- levels(Total_DataL$vertices)
  sets_v <- 1:sets
  
  # full-factorial stimulus grid
  stims <- expand.grid(curvature = 0:(length(curvature_levels)-1),
                       range = 0:1,
                       vertices = verts,
                       set = sets_v,
                       KEEP.OUT.ATTRS = FALSE,
                       stringsAsFactors = FALSE)
  stims$item <- paste0("stim", seq_len(nrow(stims)))
  
  # random intercepts for items
  stims$item_int <- rnorm(nrow(stims), 0, sd_item)
  
  # subjects
  subj_ids <- paste0("S", seq_len(Nsubj))
  subj_ints <- rnorm(Nsubj, 0, sd_subj)
  
  rows <- vector("list", Nsubj * nrow(stims) * length(size_levels))
  idx <- 1L
  
  for (i in seq_along(subj_ids)) {
    for (r in seq_len(nrow(stims))) {
      for (size in size_levels) {
        row <- stims[r,]
        mu <- 4.0 +
          subj_ints[i] +
          row$item_int +
          size_effect * (size == size_levels[2]) +   # effect for the 2nd level of size (e.g. "big")
          curvature_effect * (row$curvature)
        
        rating <- rnorm(1, mu, sd_resid)
        
        rows[[idx]] <- data.frame(
          participant = subj_ids[i],
          item = row$item,
          curvature = row$curvature,    # numeric for now; will convert to factor with correct labels below
          range = row$range,
          vertices = row$vertices,
          size = size,
          rating = rating,
          stringsAsFactors = FALSE
        )
        idx <- idx + 1L
      }
    }
  }
  
  rows_df <- do.call(rbind, rows)
  rows_df$participant <- factor(rows_df$participant)
  rows_df$item <- factor(rows_df$item)
  rows_df$size <- factor(rows_df$size, levels = levels(Total_DataL$size))
  rows_df$curvature <- factor(rows_df$curvature,
                              levels = seq_along(curvature_levels)-1,
                              labels = curvature_levels)
  rows_df$vertices <- factor(rows_df$vertices, levels = verts)
  return(rows_df)
}

# ---------------- 3) Power loop (factor-based) ----------------
power_sim <- function(Nvec = c(12,18,24,30,40,60,80), sims = 200,
                      size_effect = est_size,
                      curvature_effect = est_curvature,
                      sd_subj = SD_subj,
                      sd_item = SD_item,
                      sd_resid = SD_resid,
                      sets = 3) {
  
  results <- data.frame()
  
  size_name <- paste0("size", levels(Total_DataL$size)[2])          # e.g. "sizebig"
  curvature_name <- paste0("curvature", levels(Total_DataL$curvature)[2])  # e.g. "curvaturesmooth"
  
  
  for (N in Nvec) {
    cat(sprintf("Simulating N=%d (sims=%d) ...\n", N, sims))
    
    sim_results <- pbapply::pblapply(seq_len(sims), function(sim) {
      dsim <- simulate_once(Nsubj = N,
                            size_effect = size_effect,
                            curvature_effect = curvature_effect,
                            sd_subj = sd_subj,
                            sd_item = sd_item,
                            sd_resid = sd_resid,
                            sets = sets)
      
      
      # fit the mixed model testing main effects of size & curvature
      m <- try(lmer(rating ~ size * curvature + (1 | participant) + (1 | item), data = dsim), silent = TRUE)
      if (inherits(m, "try-error")) return(c(size_p = NA, curvature_p = NA))
      
      coefmat <- coef(summary(m))
      
      pval_size <- if (size_name %in% rownames(coefmat)) coefmat[size_name, "Pr(>|t|)"] else NA
      pval_curv <- if (curvature_name %in% rownames(coefmat)) coefmat[curvature_name, "Pr(>|t|)"] else NA
      
      return(c(size_p = pval_size, curvature_p = pval_curv))
    })
    
    sim_matrix <- do.call(rbind, sim_results)
    
    power_size <- mean(sim_matrix[, "size_p"] < 0.05, na.rm = TRUE)
    power_curvature <- mean(sim_matrix[, "curvature_p"] < 0.05, na.rm = TRUE)
    
    results <- bind_rows(results, tibble(N = N, power_size = power_size, power_curvature = power_curvature))
    cat(sprintf("  -> N=%d: power_size=%.3f, power_curvature=%.3f\n", N, power_size, power_curvature))
  }
  
  return(results)
}



# ---------------- 4) Run the simulations (adjust grid and sims as needed) ----------------

Ngrid <- c(62, 65, 68, 70)        # choose grid of sample sizes to evaluate
sims <- 300           # 300 sims is moderate; increase for precision if time allows

power_results <- power_sim(Nvec = Ngrid, sims = sims)
print(power_results)

###





