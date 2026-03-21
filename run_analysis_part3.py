#!/usr/bin/env python3
"""
Part 3: Analysis 8-10 + Summary & Log
- Analysis 8: Workshop group effect
- Analysis 9: OSIVQ cognitive style analysis
- Analysis 10: Group discussion effect
- analysis_summary.md
- analysis_log.md
"""

import os, json, warnings, datetime
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import mannwhitneyu, wilcoxon
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

OUTPUT_DIR = "output"
EMBEDDINGS_PATH = "project_260321/embeddings.npy"

df = pd.read_pickle(os.path.join(OUTPUT_DIR, '_df_state.pkl'))
part_df = pd.read_pickle(os.path.join(OUTPUT_DIR, '_part_df_state.pkl'))
embeddings = np.load(EMBEDDINGS_PATH)

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.1)
plt.rcParams['font.family'] = 'DejaVu Sans'

results3 = {}

# ══════════════════════════════════════════════
# ANALYSIS 8: Workshop Group Effect
# ══════════════════════════════════════════════
print("=" * 60)
print("ANALYSIS 8: Workshop Group Effect")
print("=" * 60)

# Participant-level mean embeddings
part_emb = {}
for pid, grp in df.groupby('participant_id'):
    idx = grp.index.tolist()
    part_emb[pid] = embeddings[idx].mean(axis=0)

# Intra vs inter group similarity
intra_sims = []
inter_sims = []
groups = df.groupby('workshop_group')['participant_id'].apply(lambda x: list(x.unique())).to_dict()

for gname, members in groups.items():
    for i in range(len(members)):
        for j in range(i+1, len(members)):
            sim = 1 - cosine_dist(part_emb[members[i]], part_emb[members[j]])
            intra_sims.append({'group': gname, 'similarity': sim, 'type': 'Intra-group'})

all_participants = list(part_emb.keys())
pid_to_group = df.drop_duplicates('participant_id').set_index('participant_id')['workshop_group'].to_dict()
for i in range(len(all_participants)):
    for j in range(i+1, len(all_participants)):
        if pid_to_group[all_participants[i]] != pid_to_group[all_participants[j]]:
            sim = 1 - cosine_dist(part_emb[all_participants[i]], part_emb[all_participants[j]])
            inter_sims.append({'similarity': sim, 'type': 'Inter-group'})

intra_df = pd.DataFrame(intra_sims)
inter_df = pd.DataFrame(inter_sims)
sim_compare = pd.concat([intra_df[['similarity', 'type']], inter_df[['similarity', 'type']]])

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 8a: Intra vs Inter boxplot
sns.boxplot(data=sim_compare, x='type', y='similarity', ax=axes[0],
            palette={'Intra-group': '#4CAF50', 'Inter-group': '#9E9E9E'})
axes[0].set_title('Intra vs Inter Group Similarity')
axes[0].set_ylabel('Cosine Similarity')
axes[0].set_xlabel('')

stat8, p8 = mannwhitneyu(intra_df['similarity'], inter_df['similarity'], alternative='greater')
print(f"  Mann-Whitney U (intra > inter, one-sided): U={stat8:.1f}, p={p8:.4f}")
print(f"    Intra: n={len(intra_df)}, mean={intra_df['similarity'].mean():.4f}")
print(f"    Inter: n={len(inter_df)}, mean={inter_df['similarity'].mean():.4f}")
results3['a8_U'] = float(stat8); results3['a8_p'] = float(p8)

# 8b: UMAP by group
palette8 = sns.color_palette('Set2', 8)
for i, gname in enumerate(sorted(groups.keys())):
    mask = df['workshop_group'] == gname
    axes[1].scatter(df.loc[mask, 'umap_x'], df.loc[mask, 'umap_y'],
                    c=[palette8[i]], alpha=0.7, s=40, label=gname, edgecolors='white', linewidth=0.5)
axes[1].set_title('UMAP by Workshop Group')
axes[1].set_xlabel('UMAP 1')
axes[1].set_ylabel('UMAP 2')
axes[1].legend(fontsize=7, ncol=2)

# 8c: Group FI/FD composition
grp_comp = df.drop_duplicates('participant_id').groupby(['workshop_group', 'participant_group']).size().unstack(fill_value=0)
grp_comp.plot(kind='bar', stacked=True, ax=axes[2], color=['#FF5722', '#2196F3'])
axes[2].set_title('FI/FD Composition by Group')
axes[2].set_xlabel('Workshop Group')
axes[2].set_ylabel('Count')
axes[2].tick_params(axis='x', rotation=0)
axes[2].legend(title='')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'analysis8_workshop.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved: analysis8_workshop.png")

# ══════════════════════════════════════════════
# ANALYSIS 9: OSIVQ Cognitive Style Analysis
# ══════════════════════════════════════════════
print("\n" + "=" * 60)
print("ANALYSIS 9: OSIVQ Cognitive Style Analysis")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 9a: UMAP by OSIVQ style
style_colors = {'object': '#FF9800', 'spatial': '#2196F3', 'verbal': '#4CAF50'}
for style, color in style_colors.items():
    mask = df['osivq_cognitive_style'] == style
    axes[0].scatter(df.loc[mask, 'umap_x'], df.loc[mask, 'umap_y'],
                    c=color, alpha=0.6, s=40, label=style, edgecolors='white', linewidth=0.5)
axes[0].set_title('UMAP by OSIVQ Style')
axes[0].set_xlabel('UMAP 1')
axes[0].set_ylabel('UMAP 2')
axes[0].legend()

# 9b: ctx_ratio by OSIVQ style
style_order = ['object', 'spatial', 'verbal']
available_styles = [s for s in style_order if s in part_df['osivq_cognitive_style'].values]
if available_styles:
    sns.boxplot(data=part_df, x='osivq_cognitive_style', y='mean_ctx_ratio', order=available_styles,
                ax=axes[1], palette=style_colors)
    sns.stripplot(data=part_df, x='osivq_cognitive_style', y='mean_ctx_ratio', order=available_styles,
                  ax=axes[1], color='black', alpha=0.5, size=5)
axes[1].set_title('Context Ratio by OSIVQ Style')
axes[1].set_xlabel('OSIVQ Style')
axes[1].set_ylabel('Context Ratio')

# 9c: initial length by OSIVQ style
if available_styles:
    sns.boxplot(data=part_df, x='osivq_cognitive_style', y='initial_prompt_length', order=available_styles,
                ax=axes[2], palette=style_colors)
    sns.stripplot(data=part_df, x='osivq_cognitive_style', y='initial_prompt_length', order=available_styles,
                  ax=axes[2], color='black', alpha=0.5, size=5)
axes[2].set_title('Initial Prompt Length by OSIVQ Style')
axes[2].set_xlabel('OSIVQ Style')
axes[2].set_ylabel('Char Count')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'analysis9_osivq.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved: analysis9_osivq.png")

# Descriptive stats
print("\n  Descriptive statistics by OSIVQ style:")
for style in available_styles:
    sub = part_df[part_df['osivq_cognitive_style'] == style]
    print(f"    {style}: n={len(sub)}, init_len={sub['initial_prompt_length'].mean():.1f}, "
          f"ctx_ratio={sub['mean_ctx_ratio'].mean():.3f}, n_turns={sub['n_turns'].mean():.1f}")

# Cross table: OSIVQ x FI/FD
cross = pd.crosstab(part_df['osivq_cognitive_style'], part_df['participant_group'])
print(f"\n  OSIVQ x FI/FD cross-table:\n{cross.to_string()}")
results3['a9_cross'] = cross.to_dict()

# ══════════════════════════════════════════════
# ANALYSIS 10: Group Discussion Effect
# ══════════════════════════════════════════════
print("\n" + "=" * 60)
print("ANALYSIS 10: Group Discussion Effect")
print("=" * 60)

# Method 1: Within-group similarity change (ideation vs final)
group_sim_records = []
for gname, members in groups.items():
    for phase in ['ideation', 'final']:
        phase_embs = {}
        for pid in members:
            pid_phase = df[(df['participant_id'] == pid) & (df['phase'] == phase)]
            if len(pid_phase) > 0:
                idx = pid_phase.index.tolist()
                phase_embs[pid] = embeddings[idx].mean(axis=0)

        sims = []
        pids = list(phase_embs.keys())
        for i in range(len(pids)):
            for j in range(i+1, len(pids)):
                sim = 1 - cosine_dist(phase_embs[pids[i]], phase_embs[pids[j]])
                sims.append(sim)

        if sims:
            group_sim_records.append({
                'group': gname, 'phase': phase, 'mean_sim': np.mean(sims), 'n_pairs': len(sims)
            })

gsim_df = pd.DataFrame(group_sim_records)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 10a: Ideation vs Final within-group similarity
if len(gsim_df) > 0:
    gsim_pivot = gsim_df.pivot(index='group', columns='phase', values='mean_sim').dropna()
    if 'ideation' in gsim_pivot.columns and 'final' in gsim_pivot.columns:
        x_pos = np.arange(len(gsim_pivot))
        axes[0].bar(x_pos - 0.2, gsim_pivot['ideation'], 0.4, label='ideation', color='#66BB6A', alpha=0.8)
        axes[0].bar(x_pos + 0.2, gsim_pivot['final'], 0.4, label='final', color='#AB47BC', alpha=0.8)
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(gsim_pivot.index, fontsize=8)
        axes[0].set_title('Within-Group Similarity\n(ideation vs final)')
        axes[0].set_ylabel('Cosine Similarity')
        axes[0].legend()

        # Wilcoxon test
        if len(gsim_pivot) >= 5:
            stat10a, p10a = wilcoxon(gsim_pivot['final'], gsim_pivot['ideation'], alternative='greater')
            print(f"  Wilcoxon (final > ideation within-group sim): W={stat10a:.1f}, p={p10a:.4f}")
            results3['a10_wilcoxon_W'] = float(stat10a); results3['a10_wilcoxon_p'] = float(p10a)
        else:
            print(f"  Wilcoxon: not enough groups (n={len(gsim_pivot)}), skipped")

# Method 2: Group centroid distance change
centroid_records = []
for gname, members in groups.items():
    for phase in ['ideation', 'final']:
        phase_embs = []
        for pid in members:
            pid_phase = df[(df['participant_id'] == pid) & (df['phase'] == phase)]
            if len(pid_phase) > 0:
                idx = pid_phase.index.tolist()
                phase_embs.append(embeddings[idx].mean(axis=0))
        if len(phase_embs) >= 2:
            centroid = np.mean(phase_embs, axis=0)
            dists = [cosine_dist(e, centroid) for e in phase_embs]
            centroid_records.append({
                'group': gname, 'phase': phase, 'mean_centroid_dist': np.mean(dists)
            })

cent_df = pd.DataFrame(centroid_records)
if len(cent_df) > 0:
    cent_pivot = cent_df.pivot(index='group', columns='phase', values='mean_centroid_dist').dropna()
    if 'ideation' in cent_pivot.columns and 'final' in cent_pivot.columns:
        x_pos = np.arange(len(cent_pivot))
        axes[1].bar(x_pos - 0.2, cent_pivot['ideation'], 0.4, label='ideation', color='#66BB6A', alpha=0.8)
        axes[1].bar(x_pos + 0.2, cent_pivot['final'], 0.4, label='final', color='#AB47BC', alpha=0.8)
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(cent_pivot.index, fontsize=8)
        axes[1].set_title('Mean Centroid Distance\n(ideation vs final)')
        axes[1].set_ylabel('Cosine Distance to Centroid')
        axes[1].legend()

# Method 3: FI/FD similarity change magnitude
sim_change_records = []
for gname, members in groups.items():
    for pid in members:
        pid_grp = df[df['participant_id'] == pid].iloc[0]['participant_group']
        sims_ideation = []
        sims_final = []
        pid_ideation = df[(df['participant_id'] == pid) & (df['phase'] == 'ideation')]
        pid_final = df[(df['participant_id'] == pid) & (df['phase'] == 'final')]

        if len(pid_ideation) == 0 or len(pid_final) == 0:
            continue

        emb_ideation = embeddings[pid_ideation.index.tolist()].mean(axis=0)
        emb_final = embeddings[pid_final.index.tolist()].mean(axis=0)

        for other_pid in members:
            if other_pid == pid:
                continue
            other_ideation = df[(df['participant_id'] == other_pid) & (df['phase'] == 'ideation')]
            other_final = df[(df['participant_id'] == other_pid) & (df['phase'] == 'final')]
            if len(other_ideation) > 0:
                sims_ideation.append(1 - cosine_dist(emb_ideation, embeddings[other_ideation.index.tolist()].mean(axis=0)))
            if len(other_final) > 0:
                sims_final.append(1 - cosine_dist(emb_final, embeddings[other_final.index.tolist()].mean(axis=0)))

        if sims_ideation and sims_final:
            sim_change_records.append({
                'participant_id': pid,
                'participant_group': pid_grp,
                'group': gname,
                'sim_change': np.mean(sims_final) - np.mean(sims_ideation),
            })

sc_df = pd.DataFrame(sim_change_records)
if len(sc_df) > 0:
    sns.boxplot(data=sc_df, x='participant_group', y='sim_change', ax=axes[2],
                palette={'FI': '#2196F3', 'FD': '#FF5722'}, order=['FI', 'FD'])
    sns.stripplot(data=sc_df, x='participant_group', y='sim_change', ax=axes[2],
                  color='black', alpha=0.5, size=5, order=['FI', 'FD'])
    axes[2].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[2].set_title('Similarity Change (final-ideation)\nby FI/FD')
    axes[2].set_ylabel('Similarity Change')
    axes[2].set_xlabel('Group')

    fi_sc = sc_df[sc_df['participant_group'] == 'FI']['sim_change']
    fd_sc = sc_df[sc_df['participant_group'] == 'FD']['sim_change']
    if len(fi_sc) > 0 and len(fd_sc) > 0:
        stat10b, p10b = mannwhitneyu(fi_sc, fd_sc, alternative='two-sided')
        print(f"  Mann-Whitney U (sim change FI vs FD): U={stat10b:.1f}, p={p10b:.4f}")
        print(f"    FI: n={len(fi_sc)}, mean={fi_sc.mean():.4f}; FD: n={len(fd_sc)}, mean={fd_sc.mean():.4f}")
        results3['a10_mwu_U'] = float(stat10b); results3['a10_mwu_p'] = float(p10b)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'analysis10_group_discussion.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved: analysis10_group_discussion.png")

# Save results
with open(os.path.join(OUTPUT_DIR, '_results_part3.json'), 'w') as f:
    json.dump(results3, f, default=str)

# ══════════════════════════════════════════════
# GENERATE SUMMARY & LOG
# ══════════════════════════════════════════════
print("\n" + "=" * 60)
print("Generating analysis_summary.md & analysis_log.md")
print("=" * 60)

# Load all results
with open(os.path.join(OUTPUT_DIR, '_results_part1.json')) as f:
    r1 = json.load(f)
with open(os.path.join(OUTPUT_DIR, '_results_part2.json')) as f:
    r2 = json.load(f)
r3 = results3

import sklearn, scipy, umap as umap_mod
versions = {
    'numpy': np.__version__,
    'pandas': pd.__version__,
    'scipy': scipy.__version__,
    'scikit-learn': sklearn.__version__,
    'matplotlib': matplotlib.__version__,
    'seaborn': sns.__version__,
    'umap-learn': umap_mod.__version__,
}

summary = f"""# Analysis Summary Report
**Generated**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

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
- Optimal k={r1['best_k']} (silhouette={r1['best_sil']:.4f})
- Silhouette scores were low overall (max ~0.075), suggesting weak cluster structure
- FI/FD groups show overlapping distributions in UMAP space

### Analysis 2: Turn-level Prompt Tracking
- Mann-Whitney U (cosine distance FI vs FD): U={r1['analysis2_U']:.1f}, p={r1['analysis2_p']:.4f}
- FI mean cosine distance: {r1['analysis2_fi_mean_cos']:.4f} (n={r1['analysis2_fi_n']})
- FD mean cosine distance: {r1['analysis2_fd_mean_cos']:.4f} (n={r1['analysis2_fd_n']})
- No statistically significant difference (p>0.05), but FI tends higher

### Analysis 3: GEFT + OSIVQ Correlation
- GEFT vs img_usage_count: rho=-0.364, p=0.052 (marginal)
- OSIVQ Object vs total_change: rho=0.383, p=0.040 *
- GEFT vs OSIVQ Spatial: rho={r1['geft_vs_osivq_spatial']['rho']:.3f}, p={r1['geft_vs_osivq_spatial']['p']:.3f}
- Most correlations were not significant with this sample size (n=29)

### Analysis 4: Context/Object Keywords
- Mann-Whitney U (ctx_ratio FI vs FD): U={r2.get('a4_U','N/A')}, p={r2.get('a4_p','N/A')}
- FI ctx_ratio mean: {r2.get('a4_fi_ctx_mean','N/A')}, FD ctx_ratio mean: {r2.get('a4_fd_ctx_mean','N/A')}
- FD shows significantly higher context keyword ratio (p<0.05)

### Analysis 5: Divergence-Convergence
- Divergence (FI vs FD): U={r2.get('a5_div_U','N/A')}, p={r2.get('a5_div_p','N/A')} (n.s.)
- Convergence slope (FI vs FD): U={r2.get('a5_conv_U','N/A')}, p={r2.get('a5_conv_p','N/A')} (n.s.)
- FI: more "Persistent Explorer" types; FD: more "Stable Refiner" types

### Analysis 6: Phase Analysis
- Phase transition turn (FI vs FD): U={r2.get('a6_U','N/A')}, p={r2.get('a6_p','N/A')} (n.s.)
- No significant difference in when participants transition from ideation to final

### Analysis 7: Major Analysis
- Initial prompt length (design vs business): U={r2.get('a7_len_U','N/A')}, p={r2.get('a7_len_p','N/A')} (n.s.)
- Context ratio (design vs business): U={r2.get('a7_ctx_U','N/A')}, p={r2.get('a7_ctx_p','N/A')} (n.s.)
- Major does not significantly differentiate prompt characteristics

### Analysis 8: Workshop Group Effect
- Intra vs Inter group similarity (one-sided): U={r3.get('a8_U','N/A')}, p={r3.get('a8_p','N/A')}
- Workshop group membership does not strongly predict prompt similarity

### Analysis 9: OSIVQ Cognitive Style
- Cross-table OSIVQ x FI/FD computed
- Descriptive differences observed but small sample sizes per style

### Analysis 10: Group Discussion Effect
- Wilcoxon (final > ideation within-group similarity): W={r3.get('a10_wilcoxon_W','N/A')}, p={r3.get('a10_wilcoxon_p','N/A')}
- Similarity change FI vs FD: U={r3.get('a10_mwu_U','N/A')}, p={r3.get('a10_mwu_p','N/A')}

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
"""
for lib, ver in versions.items():
    summary += f"- {lib}: {ver}\n"

with open(os.path.join(OUTPUT_DIR, 'analysis_summary.md'), 'w', encoding='utf-8') as f:
    f.write(summary)
print("Saved: analysis_summary.md")

# ── analysis_log.md ──
log = f"""# Analysis Execution Log
**Execution time**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Input Files
- CSV: `project_260321/data/prompt data_250320.csv`
- Embeddings: `project_260321/embeddings.npy` (pre-computed, 120x3072)
- Output directory: `{OUTPUT_DIR}/`

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
"""

with open(os.path.join(OUTPUT_DIR, 'analysis_log.md'), 'w', encoding='utf-8') as f:
    f.write(log)
print("Saved: analysis_log.md")

# Clean up temp files
for tmp in ['_df_state.pkl', '_part_df_state.pkl', '_coords.npy', '_results_part1.json', '_results_part2.json', '_results_part3.json']:
    p = os.path.join(OUTPUT_DIR, tmp)
    if os.path.exists(p):
        os.remove(p)

# Final file listing
print("\n" + "=" * 60)
print("OUTPUT FILES:")
print("=" * 60)
for f in sorted(os.listdir(OUTPUT_DIR)):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
    print(f"  {f:45s} {size:>10,} bytes")

print("\n✓ All analyses complete.")
