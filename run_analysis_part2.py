#!/usr/bin/env python3
"""
Part 2: Analysis 4-7
- Analysis 4: Context/Object keyword analysis
- Analysis 5: Divergence-Convergence
- Analysis 6: Phase analysis
- Analysis 7: Major analysis
"""

import os, re, json, warnings
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import mannwhitneyu, spearmanr, linregress
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

OUTPUT_DIR = "output"
EMBEDDINGS_PATH = "project_260321/embeddings.npy"

# Load state from Part 1
df = pd.read_pickle(os.path.join(OUTPUT_DIR, '_df_state.pkl'))
part_df = pd.read_pickle(os.path.join(OUTPUT_DIR, '_part_df_state.pkl'))
coords = np.load(os.path.join(OUTPUT_DIR, '_coords.npy'))
embeddings = np.load(EMBEDDINGS_PATH)

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.1)
plt.rcParams['font.family'] = 'DejaVu Sans'

CONTEXT_KEYWORDS = [
    '분위기','환경','공간','배경','장소','상황','맥락','컨셉','콘셉트','테마',
    '스타일','톤','무드','느낌','감성','조명','색감','컬러','색상','질감',
    '재질','소재','마감','인테리어','실내','실외','야외','도시','캠퍼스','학교',
    '교실','강의실','도서관','카페','사무실','거리','복도','건물','시설','현장',
    'atmosphere','environment','context',
    'mood','tone','style','theme','vibe','setting','background','space',
    'interior','campus','urban','lighting','color','texture','material',
]
OBJECT_KEYWORDS = [
    '버튼','화면','디스플레이','스크린','패널','모듈','센서','카메라','스피커',
    '마이크','충전기','충전','배터리','케이블','포트','USB','LED','램프','전구',
    '조명기구','키보드','마우스','태블릿','노트북','스마트폰','핸드폰','폰',
    '모니터','프린터','리모컨','스위치','손잡이','다이얼','앱','UI','아이콘',
    '메뉴','탭','슬라이더','팝업','알림','위젯','로고','라벨','태그',
    'button','screen','display','panel','module','sensor','device',
    'keyboard','mouse','tablet','laptop','smartphone','monitor','icon',
    'app','widget','menu','interface','component','element','handle',
]

def count_keywords(text, keyword_list):
    if not isinstance(text, str) or not text.strip():
        return 0
    text_lower = text.lower()
    count = 0
    for kw in keyword_list:
        count += len(re.findall(re.escape(kw.lower()), text_lower))
    return count

results2 = {}

# ══════════════════════════════════════════════
# ANALYSIS 4: Context/Object Keyword Analysis
# ══════════════════════════════════════════════
print("=" * 60)
print("ANALYSIS 4: Context/Object Keyword Analysis")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 4a: FI/FD mean keyword counts
kw_agg = df.groupby('participant_group').agg(
    ctx_mean=('ctx_count', 'mean'), obj_mean=('obj_count', 'mean'),
    ctx_std=('ctx_count', 'std'), obj_std=('obj_count', 'std')
).reset_index()

x = np.arange(2)
w = 0.35
for i, grp in enumerate(['FI', 'FD']):
    row = kw_agg[kw_agg['participant_group'] == grp].iloc[0]
    axes[0, 0].bar(i - w/2, row['ctx_mean'], w, yerr=row['ctx_std'], label='Context' if i == 0 else '', color='#4CAF50', alpha=0.8, capsize=3)
    axes[0, 0].bar(i + w/2, row['obj_mean'], w, yerr=row['obj_std'], label='Object' if i == 0 else '', color='#FF9800', alpha=0.8, capsize=3)
axes[0, 0].set_xticks([0, 1])
axes[0, 0].set_xticklabels(['FI', 'FD'])
axes[0, 0].set_title('Mean Keyword Count by FI/FD')
axes[0, 0].set_ylabel('Count')
axes[0, 0].legend()

# 4b: Keyword trend by turn
turn_kw = df.groupby(['turn_num', 'participant_group']).agg(
    ctx_mean=('ctx_count', 'mean'), obj_mean=('obj_count', 'mean')
).reset_index()
for grp, color, ls in [('FI', '#2196F3', '-'), ('FD', '#FF5722', '--')]:
    sub = turn_kw[turn_kw['participant_group'] == grp]
    axes[0, 1].plot(sub['turn_num'], sub['ctx_mean'], marker='o', color=color, linestyle=ls, markersize=4, label=f'{grp} Context')
    axes[0, 1].plot(sub['turn_num'], sub['obj_mean'], marker='s', color=color, linestyle=':', markersize=4, label=f'{grp} Object', alpha=0.6)
axes[0, 1].set_title('Keyword Trend by Turn')
axes[0, 1].set_xlabel('Turn')
axes[0, 1].set_ylabel('Mean Count')
axes[0, 1].legend(fontsize=7)

# 4c/4d: Top 15 keywords for FI and FD
for idx, (grp, ax) in enumerate([('FI', axes[1, 0]), ('FD', axes[1, 1])]):
    sub = df[df['participant_group'] == grp]
    all_kw = CONTEXT_KEYWORDS + OBJECT_KEYWORDS
    kw_counts = {}
    for kw in all_kw:
        total = sum(len(re.findall(re.escape(kw.lower()), str(t).lower())) for t in sub['prompt_combined'])
        if total > 0:
            kw_counts[kw] = total
    top15 = sorted(kw_counts.items(), key=lambda x: x[1], reverse=True)[:15]
    if top15:
        names, vals = zip(*top15)
        colors = ['#4CAF50' if n in [k.lower() for k in CONTEXT_KEYWORDS] or n in CONTEXT_KEYWORDS else '#FF9800' for n in names]
        ax.barh(range(len(names)), vals, color=colors, alpha=0.8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
    ax.set_title(f'{grp} Top 15 Keywords')
    ax.set_xlabel('Frequency')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'analysis4_keywords.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved: analysis4_keywords.png")

# Stats
fi_ctx = df[df['participant_group'] == 'FI']['ctx_ratio'].dropna()
fd_ctx = df[df['participant_group'] == 'FD']['ctx_ratio'].dropna()
if len(fi_ctx) > 0 and len(fd_ctx) > 0:
    stat4, p4 = mannwhitneyu(fi_ctx, fd_ctx, alternative='two-sided')
    r4 = 1 - (2 * stat4) / (len(fi_ctx) * len(fd_ctx))
    print(f"  Mann-Whitney U (ctx_ratio FI vs FD): U={stat4:.1f}, p={p4:.4f}, r={r4:.3f}")
    print(f"    FI: n={len(fi_ctx)}, mean={fi_ctx.mean():.4f}")
    print(f"    FD: n={len(fd_ctx)}, mean={fd_ctx.mean():.4f}")
    results2['a4_U'] = float(stat4)
    results2['a4_p'] = float(p4)
    results2['a4_fi_ctx_mean'] = float(fi_ctx.mean())
    results2['a4_fd_ctx_mean'] = float(fd_ctx.mean())

rho_gc, p_gc = spearmanr(part_df['geft_score'], part_df['mean_ctx_ratio'])
print(f"  Spearman GEFT vs ctx_ratio: rho={rho_gc:.3f}, p={p_gc:.4f}")
results2['a4_geft_ctx_rho'] = float(rho_gc)
results2['a4_geft_ctx_p'] = float(p_gc)

# ══════════════════════════════════════════════
# ANALYSIS 5: Divergence-Convergence
# ══════════════════════════════════════════════
print("\n" + "=" * 60)
print("ANALYSIS 5: Divergence-Convergence")
print("=" * 60)

div_records = []
for pid, grp in df.groupby('participant_id'):
    grp_sorted = grp.sort_values('turn_num')
    idx_list = grp_sorted.index.tolist()
    emb_p = embeddings[idx_list]

    # Divergence = cosine distance between first and last
    divergence = cosine_dist(emb_p[0], emb_p[-1]) if len(emb_p) > 1 else 0

    # Convergence = slope of cosine distances over turns
    cos_dists = []
    for i in range(1, len(emb_p)):
        cos_dists.append(cosine_dist(emb_p[i-1], emb_p[i]))

    if len(cos_dists) >= 2:
        slope, intercept, r_val, p_val, std_err = linregress(range(len(cos_dists)), cos_dists)
    else:
        slope = 0.0

    meta = grp_sorted.iloc[0]
    div_records.append({
        'participant_id': pid,
        'participant_group': meta['participant_group'],
        'divergence': divergence,
        'convergence_slope': slope,
        'n_turns': len(grp_sorted),
    })

div_df = pd.DataFrame(div_records)

# 4-type classification
div_median = div_df['divergence'].median()
slope_median = div_df['convergence_slope'].median()

def classify_type(row):
    high_div = row['divergence'] >= div_median
    neg_slope = row['convergence_slope'] <= slope_median  # negative = converging
    if high_div and not neg_slope:
        return 'Persistent Explorer'        # high div, positive slope
    if high_div and neg_slope:
        return 'Explore-then-Settle'         # high div, negative slope
    if not high_div and not neg_slope:
        return 'Gradual Diverger'            # low div, positive slope
    return 'Stable Refiner'                  # low div, negative slope

div_df['type'] = div_df.apply(classify_type, axis=1)
print("  Type distribution:")
print(div_df.groupby(['participant_group', 'type']).size().unstack(fill_value=0).to_string())

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 5a: Divergence by participant
colors5 = ['#2196F3' if g == 'FI' else '#FF5722' for g in div_df['participant_group']]
axes[0].bar(range(len(div_df)), div_df['divergence'], color=colors5, alpha=0.8)
axes[0].set_title('Divergence (First-Last Distance)')
axes[0].set_xlabel('Participant')
axes[0].set_ylabel('Cosine Distance')
axes[0].set_xticks(range(len(div_df)))
axes[0].set_xticklabels(div_df['participant_id'], rotation=90, fontsize=6)

# 5b: Convergence slope by participant
axes[1].bar(range(len(div_df)), div_df['convergence_slope'], color=colors5, alpha=0.8)
axes[1].axhline(0, color='black', linewidth=0.5)
axes[1].set_title('Convergence Slope')
axes[1].set_xlabel('Participant')
axes[1].set_ylabel('Slope')
axes[1].set_xticks(range(len(div_df)))
axes[1].set_xticklabels(div_df['participant_id'], rotation=90, fontsize=6)

# 5c: Divergence x Convergence scatter
type_colors = {'Persistent Explorer': '#E91E63', 'Explore-then-Settle': '#9C27B0',
               'Gradual Diverger': '#FF9800', 'Stable Refiner': '#4CAF50'}
type_markers = {'Persistent Explorer': 'D', 'Explore-then-Settle': 'o',
                'Gradual Diverger': 's', 'Stable Refiner': '^'}
for t in type_colors:
    sub = div_df[div_df['type'] == t]
    axes[2].scatter(sub['divergence'], sub['convergence_slope'],
                    c=type_colors[t], marker=type_markers[t], s=60, label=t, alpha=0.8, edgecolors='white')
axes[2].axhline(slope_median, color='gray', linestyle='--', alpha=0.5)
axes[2].axvline(div_median, color='gray', linestyle='--', alpha=0.5)
axes[2].set_title('Divergence x Convergence')
axes[2].set_xlabel('Divergence')
axes[2].set_ylabel('Convergence Slope')
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'analysis5_divergence.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved: analysis5_divergence.png")

# Stats
fi_div = div_df[div_df['participant_group'] == 'FI']['divergence']
fd_div = div_df[div_df['participant_group'] == 'FD']['divergence']
fi_conv = div_df[div_df['participant_group'] == 'FI']['convergence_slope']
fd_conv = div_df[div_df['participant_group'] == 'FD']['convergence_slope']

stat5d, p5d = mannwhitneyu(fi_div, fd_div, alternative='two-sided')
stat5c, p5c = mannwhitneyu(fi_conv, fd_conv, alternative='two-sided')
print(f"  Mann-Whitney U (divergence FI vs FD): U={stat5d:.1f}, p={p5d:.4f}")
print(f"    FI: n={len(fi_div)}, mean={fi_div.mean():.4f}; FD: n={len(fd_div)}, mean={fd_div.mean():.4f}")
print(f"  Mann-Whitney U (convergence FI vs FD): U={stat5c:.1f}, p={p5c:.4f}")
print(f"    FI: mean={fi_conv.mean():.4f}; FD: mean={fd_conv.mean():.4f}")
results2['a5_div_U'] = float(stat5d); results2['a5_div_p'] = float(p5d)
results2['a5_conv_U'] = float(stat5c); results2['a5_conv_p'] = float(p5c)

# ══════════════════════════════════════════════
# ANALYSIS 6: Phase Analysis
# ══════════════════════════════════════════════
print("\n" + "=" * 60)
print("ANALYSIS 6: Phase Analysis (ideation vs final)")
print("=" * 60)

# Phase-level cosine distance
phase_cos = df.dropna(subset=['cos_dist_prev']).copy()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 6a: Phase x FI/FD cosine distance boxplot
plot_data = phase_cos[['phase', 'participant_group', 'cos_dist_prev']].copy()
plot_data['group'] = plot_data['participant_group'] + ' / ' + plot_data['phase']
order = ['FI / ideation', 'FI / final', 'FD / ideation', 'FD / final']
available_order = [o for o in order if o in plot_data['group'].values]
sns.boxplot(data=plot_data, x='group', y='cos_dist_prev', order=available_order, ax=axes[0],
            palette=['#64B5F6', '#1565C0', '#FF8A65', '#D84315'])
axes[0].set_title('Cosine Distance by Phase x FI/FD')
axes[0].set_xlabel('')
axes[0].set_ylabel('Cosine Distance')
axes[0].tick_params(axis='x', rotation=15)

# 6b: Phase transition turn boxplot
transition_records = []
for pid, grp in df.groupby('participant_id'):
    grp_sorted = grp.sort_values('turn_num')
    phases = grp_sorted['phase'].tolist()
    turns = grp_sorted['turn_num'].tolist()
    transition_turn = None
    for i in range(1, len(phases)):
        if phases[i-1] == 'ideation' and phases[i] == 'final':
            transition_turn = turns[i]
            break
    if transition_turn is not None:
        transition_records.append({
            'participant_id': pid,
            'participant_group': grp_sorted.iloc[0]['participant_group'],
            'transition_turn': transition_turn
        })
trans_df = pd.DataFrame(transition_records)

if len(trans_df) > 0:
    sns.boxplot(data=trans_df, x='participant_group', y='transition_turn', ax=axes[1],
                palette={'FI': '#2196F3', 'FD': '#FF5722'}, order=['FI', 'FD'])
    sns.stripplot(data=trans_df, x='participant_group', y='transition_turn', ax=axes[1],
                  color='black', alpha=0.5, size=5, order=['FI', 'FD'])
    axes[1].set_title('Phase Transition Turn (ideation->final)')
    axes[1].set_xlabel('Group')
    axes[1].set_ylabel('Turn Number')

    fi_trans = trans_df[trans_df['participant_group'] == 'FI']['transition_turn']
    fd_trans = trans_df[trans_df['participant_group'] == 'FD']['transition_turn']
    if len(fi_trans) > 0 and len(fd_trans) > 0:
        stat6, p6 = mannwhitneyu(fi_trans, fd_trans, alternative='two-sided')
        print(f"  Mann-Whitney U (transition turn FI vs FD): U={stat6:.1f}, p={p6:.4f}")
        print(f"    FI: n={len(fi_trans)}, mean={fi_trans.mean():.2f}; FD: n={len(fd_trans)}, mean={fd_trans.mean():.2f}")
        results2['a6_U'] = float(stat6); results2['a6_p'] = float(p6)
else:
    axes[1].set_title('No phase transitions found')

# 6c: UMAP phase mapping
for phase, marker, color in [('ideation', 'o', '#66BB6A'), ('final', 's', '#AB47BC')]:
    mask = df['phase'] == phase
    axes[2].scatter(df.loc[mask, 'umap_x'], df.loc[mask, 'umap_y'],
                    marker=marker, c=color, alpha=0.6, s=40, label=phase, edgecolors='white', linewidth=0.5)
axes[2].set_title('UMAP by Phase')
axes[2].set_xlabel('UMAP 1')
axes[2].set_ylabel('UMAP 2')
axes[2].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'analysis6_phase.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved: analysis6_phase.png")

# ══════════════════════════════════════════════
# ANALYSIS 7: Major Analysis (design vs business)
# ══════════════════════════════════════════════
print("\n" + "=" * 60)
print("ANALYSIS 7: Major Analysis (design vs business)")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 7a: UMAP by major
for major, marker, color in [('design', 'o', '#26A69A'), ('business', 's', '#EF5350')]:
    mask = df['major'] == major
    axes[0].scatter(df.loc[mask, 'umap_x'], df.loc[mask, 'umap_y'],
                    marker=marker, c=color, alpha=0.6, s=40, label=major, edgecolors='white', linewidth=0.5)
axes[0].set_title('UMAP by Major')
axes[0].set_xlabel('UMAP 1')
axes[0].set_ylabel('UMAP 2')
axes[0].legend()

# 7b: Major x Cognitive style ctx_ratio boxplot
part_df_ext = part_df.copy()
part_df_ext['group_4'] = part_df_ext['major'] + ' / ' + part_df_ext['participant_group']
order7 = ['design / FI', 'design / FD', 'business / FI', 'business / FD']
available7 = [o for o in order7 if o in part_df_ext['group_4'].values]
if available7:
    sns.boxplot(data=part_df_ext, x='group_4', y='mean_ctx_ratio', order=available7, ax=axes[1],
                palette=['#64B5F6', '#FF8A65', '#1565C0', '#D84315'])
    sns.stripplot(data=part_df_ext, x='group_4', y='mean_ctx_ratio', order=available7, ax=axes[1],
                  color='black', alpha=0.5, size=5)
axes[1].set_title('Context Ratio by Major x FI/FD')
axes[1].set_xlabel('')
axes[1].set_ylabel('Context Ratio')
axes[1].tick_params(axis='x', rotation=15)

# 7c: Initial prompt length by major
sns.boxplot(data=part_df, x='major', y='initial_prompt_length', ax=axes[2],
            palette={'design': '#26A69A', 'business': '#EF5350'}, order=['design', 'business'])
sns.stripplot(data=part_df, x='major', y='initial_prompt_length', ax=axes[2],
              color='black', alpha=0.5, size=5, order=['design', 'business'])
axes[2].set_title('Initial Prompt Length by Major')
axes[2].set_xlabel('Major')
axes[2].set_ylabel('Char Count')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'analysis7_major.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved: analysis7_major.png")

# Stats
design_len = part_df[part_df['major'] == 'design']['initial_prompt_length']
biz_len = part_df[part_df['major'] == 'business']['initial_prompt_length']
stat7a, p7a = mannwhitneyu(design_len, biz_len, alternative='two-sided')
print(f"  Mann-Whitney U (init length design vs business): U={stat7a:.1f}, p={p7a:.4f}")
print(f"    design: n={len(design_len)}, mean={design_len.mean():.1f}; business: n={len(biz_len)}, mean={biz_len.mean():.1f}")

design_ctx = part_df[part_df['major'] == 'design']['mean_ctx_ratio'].dropna()
biz_ctx = part_df[part_df['major'] == 'business']['mean_ctx_ratio'].dropna()
if len(design_ctx) > 0 and len(biz_ctx) > 0:
    stat7b, p7b = mannwhitneyu(design_ctx, biz_ctx, alternative='two-sided')
    print(f"  Mann-Whitney U (ctx_ratio design vs business): U={stat7b:.1f}, p={p7b:.4f}")
    results2['a7_ctx_U'] = float(stat7b); results2['a7_ctx_p'] = float(p7b)

results2['a7_len_U'] = float(stat7a); results2['a7_len_p'] = float(p7a)

# Save results
with open(os.path.join(OUTPUT_DIR, '_results_part2.json'), 'w') as f:
    json.dump(results2, f)

print("\n✓ Part 2 complete.")
