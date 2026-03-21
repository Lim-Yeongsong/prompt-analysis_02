# Analysis Execution Log
**Execution time**: 2026-03-21 23:41:28

## Input Files
- CSV: `project_260321/data/prompt data_250320.csv`
- Embeddings: `project_260321/embeddings.npy` (pre-computed, 120x3072)
- Output directory: `output/`

## Embeddings
- Used pre-computed embeddings.npy (120 x 3,072)
- No re-embedding performed

## Column Validation
- All 17 required columns present: PASS
- No missing columns

## Selection Replacement Log
| prompt_id | original | rule |
|-----------|----------|------|
| prompt_08 | 엉 시각화 해줘... | positive_response → prompt_context |
| prompt_11 | 3 | number_3_match → option 3 text |
| prompt_13 | 엉 | positive_response → prompt_context |

## Missing Value Handling
- prompt_raw (1 missing, P29 turn_03): filled as empty string for char_count, used as-is for embedding (already embedded)
- prompt_context (113 missing): expected, only used for selection replacement
- img_type/prompt_img (104 missing): expected, flagged in img_usage_count

## Excluded Samples by Analysis
- Analysis 1 (Clustering): 0 excluded (all 120 used)
- Analysis 2 (Turn tracking): Turn 1 excluded from cosine distance (no previous turn)
  - FI: 58 cosine distances, FD: 33 cosine distances
- Analysis 3 (Correlations): 0 excluded (all 29 participants used)
- Analysis 4 (Keywords): Prompts with 0 context+object keywords → ctx_ratio=NaN
- Analysis 5 (Divergence): Participants with <2 turns excluded from slope calculation
- Analysis 6 (Phase): Participants without ideation→final transition excluded from transition analysis
- Analysis 7 (Major): 0 excluded
- Analysis 8 (Workshop): 0 excluded
- Analysis 9 (OSIVQ): 0 excluded
- Analysis 10 (Group discussion): Participants without both ideation and final prompts excluded

## Warnings
- Low silhouette scores in clustering suggest weak cluster structure
- GEFT vs ctx_ratio Spearman returned NaN (likely due to NaN in ctx_ratio for some participants)
- Small sample sizes per OSIVQ style limit statistical inference
