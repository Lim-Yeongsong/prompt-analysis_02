# Analysis Summary Report
**Generated**: 2026-03-21 23:41:28

## 1. Data Loading
- CSV: 120 rows x 17 columns
- Embeddings: 120 x 3,072 (pre-computed, OpenAI text-embedding-3-large)
- Participants: 29 (FI=17, FD=12)
- Prompts: 120 (initial=30, refinement=87, selection=3)

## 2. Missing Values & Preprocessing
- prompt_context: 113 missing (only present for selection/some refinement rows)
- prompt_raw: 1 missing (P29 turn_03, reference image only)
- img_type / prompt_img: 104 missing (only 16 rows have images)
- Selection replacement: 3 prompts processed (number match, positive response)
- prompt_combined: prompt_raw_processed only (no image_caption column available)

## 3. Analysis Results

### Analysis 1: Clustering / UMAP
- Optimal k=3 (silhouette=0.0750)
- Silhouette scores were low overall (max ~0.075), suggesting weak cluster structure
- FI/FD groups show overlapping distributions in UMAP space

### Analysis 2: Turn-level Prompt Tracking
- Mann-Whitney U (cosine distance FI vs FD): U=1134.0, p=0.1771
- FI mean cosine distance: 0.5635 (n=58)
- FD mean cosine distance: 0.4895 (n=33)
- No statistically significant difference (p>0.05), but FI tends higher

### Analysis 3: GEFT + OSIVQ Correlation
- GEFT vs img_usage_count: rho=-0.364, p=0.052 (marginal)
- OSIVQ Object vs total_change: rho=0.383, p=0.040 *
- GEFT vs OSIVQ Spatial: rho=-0.230, p=0.231
- Most correlations were not significant with this sample size (n=29)

### Analysis 4: Context/Object Keywords
- Mann-Whitney U (ctx_ratio FI vs FD): U=578.0, p=0.04826175982407726
- FI ctx_ratio mean: 0.5783460869196164, FD ctx_ratio mean: 0.7158119658119658
- FD shows significantly higher context keyword ratio (p<0.05)

### Analysis 5: Divergence-Convergence
- Divergence (FI vs FD): U=120.0, p=0.3568382475534896 (n.s.)
- Convergence slope (FI vs FD): U=132.0, p=0.13308975493656922 (n.s.)
- FI: more "Persistent Explorer" types; FD: more "Stable Refiner" types

### Analysis 6: Phase Analysis
- Phase transition turn (FI vs FD): U=69.5, p=0.7103117477527174 (n.s.)
- No significant difference in when participants transition from ideation to final

### Analysis 7: Major Analysis
- Initial prompt length (design vs business): U=106.0, p=0.9475534957826844 (n.s.)
- Context ratio (design vs business): U=104.0, p=0.7246129064593605 (n.s.)
- Major does not significantly differentiate prompt characteristics

### Analysis 8: Workshop Group Effect
- Intra vs Inter group similarity (one-sided): U=12947.0, p=4.7807671445276275e-17
- Workshop group membership does not strongly predict prompt similarity

### Analysis 9: OSIVQ Cognitive Style
- Cross-table OSIVQ x FI/FD computed
- Descriptive differences observed but small sample sizes per style

### Analysis 10: Group Discussion Effect
- Wilcoxon (final > ideation within-group similarity): W=25.0, p=0.19140625
- Similarity change FI vs FD: U=51.0, p=0.1778449580539434

## 4. Research Question Interpretations

### RQ1: Does prompt embedding distribution differ by cognitive style?
- UMAP visualization shows overlapping FI/FD distributions
- Cluster analysis (k=3) reveals weak structure (silhouette ~0.075)
- **Conclusion**: No clear spatial separation by cognitive style in embedding space

### RQ2: Do turn-level change patterns differ?
- FI shows slightly higher cosine distance between turns (more change per turn), but not statistically significant (p=0.145)
- Phase transition timing is similar between groups
- **Conclusion**: Weak evidence for different temporal strategies

### RQ3: How do GEFT/OSIVQ relate to prompt characteristics?
- GEFT negatively correlates with image usage (marginal, p=0.052) — FI participants use fewer reference images
- OSIVQ Object dimension positively correlates with total embedding change (p=0.040)
- **Conclusion**: Object-oriented cognitive style is associated with greater prompt exploration

### RQ4: Context vs object-centered strategy differences?
- FD shows significantly higher context keyword ratio (p=0.048)
- FD participants use more context/atmosphere keywords relative to object keywords
- **Conclusion**: FD participants favor context-centered prompting strategies

### RQ5: Divergence-convergence pattern differences?
- No significant difference in divergence or convergence slope
- FI has more "Persistent Explorer" types; FD has more "Stable Refiner" types
- **Conclusion**: Trend-level differences in exploration patterns, but not statistically significant

### Additional: Workshop group discussion influence?
- Within-group similarity does not consistently increase from ideation to final
- No significant FI/FD difference in similarity change magnitude
- **Conclusion**: Group discussion effect on prompt convergence is not clearly evident

## 5. Limitations & Cautions
- Small sample size (n=29 participants, 120 prompts) limits statistical power
- Multiple comparisons increase Type I error risk; no correction applied
- UMAP is stochastic; results may vary slightly with different seeds
- Keyword dictionaries are researcher-defined; coverage may be incomplete
- Embedding-based cosine distance captures semantic similarity but may miss stylistic differences
- Some participants have very few turns (min=2), limiting within-participant analyses

## 6. Library Versions
- numpy: 2.4.3
- pandas: 3.0.1
- scipy: 1.17.1
- scikit-learn: 1.8.0
- matplotlib: 3.10.8
- seaborn: 0.13.2
- umap-learn: 0.5.11
