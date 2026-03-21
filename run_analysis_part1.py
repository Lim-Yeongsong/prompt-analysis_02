#!/usr/bin/env python3
"""
Part 1: Preprocessing + Analysis 1-3
- Data validation & preprocessing
- Participant-level aggregation
- Analysis 1: Clustering / UMAP
- Analysis 2: Turn-level prompt tracking
- Analysis 3: GEFT + OSIVQ correlation
"""

import os, re, warnings, datetime
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

warnings.filterwarnings('ignore')

# ── Paths ──
INPUT_CSV = "project_260321/data/prompt data_250320.csv"
EMBEDDINGS_PATH = "project_260321/embeddings.npy"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Font setup (Korean fallback to English) ──
def setup_font():
    korean_fonts = [f.name for f in fm.fontManager.ttflist if any(k in f.name.lower() for k in ['nanum','malgun','noto sans cjk','gulim'])]
    if korean_fonts:
        plt.rcParams['font.family'] = korean_fonts[0]
        plt.rcParams['axes.unicode_minus'] = False
        return True
    plt.rcParams['font.family'] = 'DejaVu Sans'
    return False

HAS_KOREAN_FONT = setup_font()
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.1)

# ══════════════════════════════════════════════
# 0. DATA LOADING & VALIDATION
# ══════════════════════════════════════════════
print("=" * 60)
print("STEP 0: Data Loading & Validation")
print("=" * 60)

df = pd.read_csv(INPUT_CSV)
print(f"Loaded CSV: {df.shape[0]} rows x {df.shape[1]} columns")

REQUIRED_COLS = [
    'prompt_id','turn','participant_id','major','workshop_group',
    'geft_score','participant_group','osivq_object_pct','osivq_spatial_pct',
    'osivq_verbal_pct','osivq_cognitive_style','phase','prompt_type',
    'prompt_context','prompt_raw','img_type','prompt_img'
]
missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")
print("All required columns present.")

# Load embeddings
embeddings = np.load(EMBEDDINGS_PATH)
print(f"Loaded embeddings: {embeddings.shape}")
if embeddings.shape[0] != len(df):
    raise ValueError(f"Embedding count ({embeddings.shape[0]}) != CSV rows ({len(df)})")
print("Embedding count matches CSV rows.")

# ══════════════════════════════════════════════
# 1. PREPROCESSING
# ══════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 1: Preprocessing")
print("=" * 60)

# Turn number extraction
df['turn_num'] = df['turn'].str.extract(r'(\d+)').astype(int)

# 1-1. Selection replacement
selection_log = []

def replace_selection(row):
    if row['prompt_type'] != 'selection':
        return row['prompt_raw']
    raw = str(row['prompt_raw']).strip() if pd.notna(row['prompt_raw']) else ''
    context = str(row['prompt_context']) if pd.notna(row['prompt_context']) else ''
    original = raw

    # Case: number only -> extract that option from context
    if re.match(r'^\d+$', raw):
        num = int(raw)
        # Try to find numbered options in context
        lines = context.split('\n')
        for line in lines:
            m = re.match(r'[^\d]*(\d+)[^\w]*(.*)', line.strip())
            if m and int(m.group(1)) == num:
                replaced = m.group(2).strip()
                selection_log.append({'prompt_id': row['prompt_id'], 'original': original, 'replaced': replaced, 'rule': f'number_{num}_match'})
                return replaced
        # Fallback: use context itself
        selection_log.append({'prompt_id': row['prompt_id'], 'original': original, 'replaced': context, 'rule': 'number_fallback_context'})
        return context

    # Case: positive response
    positive_patterns = r'^(yes|yeah|good|okay|ok|sure|go|엉|응|네|예|좋아|마음에|괜찮|그래|해줘|ㅇㅇ|넹|웅)'
    if re.match(positive_patterns, raw, re.IGNORECASE):
        replaced = context if context else raw
        selection_log.append({'prompt_id': row['prompt_id'], 'original': original, 'replaced': replaced, 'rule': 'positive_response'})
        return replaced

    selection_log.append({'prompt_id': row['prompt_id'], 'original': original, 'replaced': raw, 'rule': 'no_change'})
    return raw

df['prompt_raw_processed'] = df.apply(replace_selection, axis=1)
sel_log_df = pd.DataFrame(selection_log)
if len(sel_log_df) > 0:
    sel_log_df.to_csv(os.path.join(OUTPUT_DIR, 'selection_replacement_log.csv'), index=False)
    print(f"Selection replacement log saved: {len(sel_log_df)} entries")
    print(sel_log_df.to_string())

# 1-2. Derived variables
df['char_count'] = df['prompt_raw_processed'].fillna('').str.len()
df['char_count_no_space'] = df['prompt_raw_processed'].fillna('').str.replace(' ', '', regex=False).str.len()

# 1-3. Text combination (prompt_combined)
if 'image_caption' in df.columns:
    df['prompt_combined'] = df['prompt_raw_processed'].fillna('') + ' ' + df['image_caption'].fillna('')
    combine_method = "prompt_raw_processed + image_caption"
else:
    df['prompt_combined'] = df['prompt_raw_processed'].fillna('')
    combine_method = "prompt_raw_processed only (no image_caption column)"
df['prompt_combined'] = df['prompt_combined'].str.strip()
print(f"prompt_combined method: {combine_method}")

# 1-4. UMAP dimension reduction
print("Running UMAP...")
reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
coords = reducer.fit_transform(embeddings)
df['umap_x'] = coords[:, 0]
df['umap_y'] = coords[:, 1]
print(f"UMAP complete: {coords.shape}")

# 1-5. Keyword dictionaries and classification
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
        pattern = re.escape(kw.lower())
        count += len(re.findall(pattern, text_lower))
    return count

df['ctx_count'] = df['prompt_combined'].apply(lambda x: count_keywords(x, CONTEXT_KEYWORDS))
df['obj_count'] = df['prompt_combined'].apply(lambda x: count_keywords(x, OBJECT_KEYWORDS))
df['ctx_ratio'] = df.apply(lambda r: r['ctx_count'] / (r['ctx_count'] + r['obj_count'])
                           if (r['ctx_count'] + r['obj_count']) > 0 else np.nan, axis=1)
print(f"Keyword counts: ctx_count mean={df['ctx_count'].mean():.2f}, obj_count mean={df['obj_count'].mean():.2f}")

# Save preprocessed CSV
df.to_csv(os.path.join(OUTPUT_DIR, 'prompt_preprocessed.csv'), index=False, encoding='utf-8-sig')
print(f"Saved: prompt_preprocessed.csv")

# Save embeddings copy
np.save(os.path.join(OUTPUT_DIR, 'embeddings.npy'), embeddings)
print(f"Saved: embeddings.npy")

# ══════════════════════════════════════════════
# PARTICIPANT-LEVEL AGGREGATION
# ══════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP: Participant-level aggregation")
print("=" * 60)

def compute_participant_features(df, embeddings):
    records = []
    for pid, grp in df.groupby('participant_id'):
        grp_sorted = grp.sort_values('turn_num')
        idx = grp_sorted.index.tolist()
        emb_p = embeddings[idx]

        # Initial prompt length
        initial = grp_sorted[grp_sorted['prompt_type'] == 'initial']
        init_len = initial['char_count'].values[0] if len(initial) > 0 else grp_sorted.iloc[0]['char_count']

        # Turn count
        n_turns = len(grp_sorted)

        # Cosine distances between consecutive turns
        cos_dists = []
        for i in range(1, len(emb_p)):
            d = cosine_dist(emb_p[i-1], emb_p[i])
            cos_dists.append(d)

        total_change = sum(cos_dists) if cos_dists else 0
        first_last_dist = cosine_dist(emb_p[0], emb_p[-1]) if len(emb_p) > 1 else 0

        # Image usage count
        img_count = grp_sorted['prompt_img'].notna().sum()

        # Ideation ratio
        ideation_ratio = (grp_sorted['phase'] == 'ideation').mean()

        # Additional
        mean_char = grp_sorted['char_count'].mean()
        ctx_ratio_mean = grp_sorted['ctx_ratio'].mean()

        meta = grp_sorted.iloc[0]
        records.append({
            'participant_id': pid,
            'participant_group': meta['participant_group'],
            'major': meta['major'],
            'workshop_group': meta['workshop_group'],
            'geft_score': meta['geft_score'],
            'osivq_object_pct': meta['osivq_object_pct'],
            'osivq_spatial_pct': meta['osivq_spatial_pct'],
            'osivq_verbal_pct': meta['osivq_verbal_pct'],
            'osivq_cognitive_style': meta['osivq_cognitive_style'],
            'initial_prompt_length': init_len,
            'n_turns': n_turns,
            'total_change': total_change,
            'first_last_distance': first_last_dist,
            'img_usage_count': img_count,
            'ideation_ratio': ideation_ratio,
            'mean_char_count': mean_char,
            'mean_ctx_ratio': ctx_ratio_mean,
        })
    return pd.DataFrame(records)

part_df = compute_participant_features(df, embeddings)
part_df.to_csv(os.path.join(OUTPUT_DIR, 'participant_analysis.csv'), index=False, encoding='utf-8-sig')
print(f"Saved: participant_analysis.csv ({len(part_df)} participants)")
print(part_df[['participant_id','participant_group','initial_prompt_length','n_turns','total_change','first_last_distance','img_usage_count','ideation_ratio']].to_string())

# ══════════════════════════════════════════════
# ANALYSIS 1: Clustering / UMAP
# ══════════════════════════════════════════════
print("\n" + "=" * 60)
print("ANALYSIS 1: Clustering / UMAP Visualization")
print("=" * 60)

sil_scores = {}
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)
    sil = silhouette_score(embeddings, labels)
    sil_scores[k] = sil
    print(f"  k={k}: silhouette={sil:.4f}")

best_k = max(sil_scores, key=sil_scores.get)
print(f"  Best k={best_k} (silhouette={sil_scores[best_k]:.4f})")

km_best = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['cluster'] = km_best.fit_predict(embeddings)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1a: FI/FD UMAP
for grp, marker, color in [('FI', 'o', '#2196F3'), ('FD', 's', '#FF5722')]:
    mask = df['participant_group'] == grp
    axes[0].scatter(df.loc[mask, 'umap_x'], df.loc[mask, 'umap_y'],
                    c=color, marker=marker, alpha=0.7, s=40, label=grp, edgecolors='white', linewidth=0.5)
axes[0].set_title('UMAP by FI/FD')
axes[0].set_xlabel('UMAP 1')
axes[0].set_ylabel('UMAP 2')
axes[0].legend()

# 1b: Cluster UMAP
palette = sns.color_palette('Set2', best_k)
for c in range(best_k):
    mask = df['cluster'] == c
    axes[1].scatter(df.loc[mask, 'umap_x'], df.loc[mask, 'umap_y'],
                    c=[palette[c]], alpha=0.7, s=40, label=f'C{c}', edgecolors='white', linewidth=0.5)
axes[1].set_title(f'UMAP by Cluster (k={best_k})')
axes[1].set_xlabel('UMAP 1')
axes[1].set_ylabel('UMAP 2')
axes[1].legend()

# 1c: Silhouette scores
axes[2].bar(sil_scores.keys(), sil_scores.values(), color='steelblue')
axes[2].axvline(best_k, color='red', linestyle='--', label=f'Best k={best_k}')
axes[2].set_xlabel('k')
axes[2].set_ylabel('Silhouette Score')
axes[2].set_title('Silhouette Score by k')
axes[2].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'analysis1_clustering.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved: analysis1_clustering.png")

# ══════════════════════════════════════════════
# ANALYSIS 2: Turn-level Prompt Tracking
# ══════════════════════════════════════════════
print("\n" + "=" * 60)
print("ANALYSIS 2: Turn-level Prompt Tracking")
print("=" * 60)

# Compute per-prompt cosine distance to previous turn (same participant)
df['cos_dist_prev'] = np.nan
for pid, grp in df.groupby('participant_id'):
    grp_sorted = grp.sort_values('turn_num')
    idx_list = grp_sorted.index.tolist()
    for i in range(1, len(idx_list)):
        d = cosine_dist(embeddings[idx_list[i-1]], embeddings[idx_list[i]])
        df.loc[idx_list[i], 'cos_dist_prev'] = d

# Turn-level aggregation
turn_agg = df.groupby(['turn_num', 'participant_group']).agg(
    mean_char=('char_count', 'mean'),
    n_participants=('participant_id', 'nunique'),
    mean_cos_dist=('cos_dist_prev', 'mean'),
).reset_index()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 2a: Char count by turn
for grp, color in [('FI', '#2196F3'), ('FD', '#FF5722')]:
    sub = turn_agg[turn_agg['participant_group'] == grp]
    axes[0, 0].plot(sub['turn_num'], sub['mean_char'], marker='o', color=color, label=grp, markersize=4)
axes[0, 0].set_title('Mean Char Count by Turn')
axes[0, 0].set_xlabel('Turn')
axes[0, 0].set_ylabel('Char Count')
axes[0, 0].legend()

# 2b: Participants by turn
for grp, color in [('FI', '#2196F3'), ('FD', '#FF5722')]:
    sub = turn_agg[turn_agg['participant_group'] == grp]
    axes[0, 1].bar(sub['turn_num'] + (0.2 if grp == 'FD' else -0.2), sub['n_participants'],
                   width=0.4, color=color, label=grp, alpha=0.8)
axes[0, 1].set_title('Participants by Turn')
axes[0, 1].set_xlabel('Turn')
axes[0, 1].set_ylabel('Count')
axes[0, 1].legend()

# 2c: Cosine distance by turn
for grp, color in [('FI', '#2196F3'), ('FD', '#FF5722')]:
    sub = turn_agg[turn_agg['participant_group'] == grp]
    sub = sub.dropna(subset=['mean_cos_dist'])
    axes[1, 0].plot(sub['turn_num'], sub['mean_cos_dist'], marker='o', color=color, label=grp, markersize=4)
axes[1, 0].set_title('Mean Cosine Distance by Turn')
axes[1, 0].set_xlabel('Turn')
axes[1, 0].set_ylabel('Cosine Distance')
axes[1, 0].legend()

# 2d: Prompt type distribution
type_dist = df.groupby(['turn_num', 'prompt_type']).size().unstack(fill_value=0)
type_pct = type_dist.div(type_dist.sum(axis=1), axis=0) * 100
type_pct.plot(kind='bar', stacked=True, ax=axes[1, 1], colormap='Set2', width=0.8)
axes[1, 1].set_title('Prompt Type Distribution by Turn (%)')
axes[1, 1].set_xlabel('Turn')
axes[1, 1].set_ylabel('%')
axes[1, 1].legend(fontsize=8)
axes[1, 1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'analysis2_turn.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved: analysis2_turn.png")

# Stats: Mann-Whitney U for cosine distance
fi_cos = df[df['participant_group'] == 'FI']['cos_dist_prev'].dropna()
fd_cos = df[df['participant_group'] == 'FD']['cos_dist_prev'].dropna()
stat, p = mannwhitneyu(fi_cos, fd_cos, alternative='two-sided')
r_effect = 1 - (2 * stat) / (len(fi_cos) * len(fd_cos))
print(f"  Mann-Whitney U (cosine dist FI vs FD): U={stat:.1f}, p={p:.4f}, r={r_effect:.3f}")
print(f"    FI: n={len(fi_cos)}, mean={fi_cos.mean():.4f}")
print(f"    FD: n={len(fd_cos)}, mean={fd_cos.mean():.4f}")

# ══════════════════════════════════════════════
# ANALYSIS 3: GEFT + OSIVQ Correlation
# ══════════════════════════════════════════════
print("\n" + "=" * 60)
print("ANALYSIS 3: GEFT + OSIVQ Correlation")
print("=" * 60)

features = ['initial_prompt_length', 'n_turns', 'total_change',
            'first_last_distance', 'img_usage_count', 'ideation_ratio']
feat_labels = ['Init Length', 'N Turns', 'Total Change',
               'First-Last Dist', 'Img Usage', 'Ideation Ratio']

# GEFT correlations
print("\n  GEFT Spearman correlations:")
geft_corrs = []
for f in features:
    rho, p = spearmanr(part_df['geft_score'], part_df[f])
    geft_corrs.append((f, rho, p))
    sig = '*' if p < 0.05 else ''
    print(f"    GEFT vs {f}: rho={rho:.3f}, p={p:.4f} {sig}")

# OSIVQ correlations
print("\n  OSIVQ Spearman correlations:")
osivq_dims = ['osivq_object_pct', 'osivq_spatial_pct', 'osivq_verbal_pct']
osivq_labels = ['Object', 'Spatial', 'Verbal']
osivq_corrs = []
for od, ol in zip(osivq_dims, osivq_labels):
    for f in features:
        rho, p = spearmanr(part_df[od], part_df[f])
        osivq_corrs.append((ol, f, rho, p))
        sig = '*' if p < 0.05 else ''
        print(f"    {ol} vs {f}: rho={rho:.3f}, p={p:.4f} {sig}")

# GEFT vs OSIVQ spatial
rho_gs, p_gs = spearmanr(part_df['geft_score'], part_df['osivq_spatial_pct'])
print(f"\n  GEFT vs OSIVQ Spatial: rho={rho_gs:.3f}, p={p_gs:.4f}")

# Visualization
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Row 1: GEFT scatter plots (6)
for i, (f, fl) in enumerate(zip(features, feat_labels)):
    ax = axes[0, i] if i < 4 else axes[1, i - 4]
    colors = ['#2196F3' if g == 'FI' else '#FF5722' for g in part_df['participant_group']]
    ax.scatter(part_df['geft_score'], part_df[f], c=colors, alpha=0.7, s=50, edgecolors='white')
    rho, p = spearmanr(part_df['geft_score'], part_df[f])
    ax.set_title(f'GEFT vs {fl}\nrho={rho:.3f}, p={p:.3f}', fontsize=10)
    ax.set_xlabel('GEFT Score')
    ax.set_ylabel(fl)

# Row 2 remaining: OSIVQ spatial scatter + GEFT vs OSIVQ spatial
ax_sp = axes[1, 2]
colors = ['#2196F3' if g == 'FI' else '#FF5722' for g in part_df['participant_group']]
ax_sp.scatter(part_df['osivq_spatial_pct'], part_df['first_last_distance'], c=colors, alpha=0.7, s=50, edgecolors='white')
rho_s, p_s = spearmanr(part_df['osivq_spatial_pct'], part_df['first_last_distance'])
ax_sp.set_title(f'OSIVQ Spatial vs First-Last Dist\nrho={rho_s:.3f}, p={p_s:.3f}', fontsize=10)
ax_sp.set_xlabel('OSIVQ Spatial %')
ax_sp.set_ylabel('First-Last Distance')

ax_gs = axes[1, 3]
ax_gs.scatter(part_df['geft_score'], part_df['osivq_spatial_pct'], c=colors, alpha=0.7, s=50, edgecolors='white')
ax_gs.set_title(f'GEFT vs OSIVQ Spatial\nrho={rho_gs:.3f}, p={p_gs:.3f}', fontsize=10)
ax_gs.set_xlabel('GEFT Score')
ax_gs.set_ylabel('OSIVQ Spatial %')

# Legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0],[0], marker='o', color='w', markerfacecolor='#2196F3', markersize=8, label='FI'),
                   Line2D([0],[0], marker='o', color='w', markerfacecolor='#FF5722', markersize=8, label='FD')]
fig.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'analysis3_geft_osivq.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved: analysis3_geft_osivq.png")

# Save intermediate state for Part 2
df.to_pickle(os.path.join(OUTPUT_DIR, '_df_state.pkl'))
part_df.to_pickle(os.path.join(OUTPUT_DIR, '_part_df_state.pkl'))
np.save(os.path.join(OUTPUT_DIR, '_coords.npy'), coords)

# Save analysis results for summary
import json
results = {
    'best_k': int(best_k),
    'best_sil': float(sil_scores[best_k]),
    'sil_scores': {str(k): float(v) for k, v in sil_scores.items()},
    'analysis2_U': float(stat),
    'analysis2_p': float(p),
    'analysis2_fi_mean_cos': float(fi_cos.mean()),
    'analysis2_fd_mean_cos': float(fd_cos.mean()),
    'analysis2_fi_n': int(len(fi_cos)),
    'analysis2_fd_n': int(len(fd_cos)),
    'geft_corrs': [(f, float(r), float(p)) for f, r, p in geft_corrs],
    'geft_vs_osivq_spatial': {'rho': float(rho_gs), 'p': float(p_gs)},
}
with open(os.path.join(OUTPUT_DIR, '_results_part1.json'), 'w') as f:
    json.dump(results, f)

print("\n✓ Part 1 complete.")
