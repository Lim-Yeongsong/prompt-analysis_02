#!/usr/bin/env python3
"""추가 분석: Axis 1 키워드 수/비율 + Phase별 발산·수렴·키워드 차이"""
import pickle, json, os
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid"); sns.set_context("paper", font_scale=1.1)
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False
FI_COLOR, FD_COLOR = '#2196F3', '#FF5722'
IDE_COLOR, FIN_COLOR = '#66BB6A', '#AB47BC'

def effect_size_r(U, n1, n2):
    return 1 - (2 * U) / (n1 * n2)

def mwu(a, b, var_name, label_a='FI', label_b='FD'):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    U, p = mannwhitneyu(a, b, alternative='two-sided')
    r = effect_size_r(U, len(a), len(b))
    sig = '***' if p<.001 else '**' if p<.01 else '*' if p<.05 else 'n.s.'
    return {'test':'MWU', 'var':var_name, 'label_a':label_a, 'label_b':label_b,
            'n_a':len(a), 'M_a':float(np.mean(a)), 'SD_a':float(np.std(a, ddof=1)) if len(a)>1 else 0,
            'n_b':len(b), 'M_b':float(np.mean(b)), 'SD_b':float(np.std(b, ddof=1)) if len(b)>1 else 0,
            'U':float(U), 'p':float(p), 'r':float(r), 'sig':sig}

def spear(x, y, x_label, y_label):
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    rho, p = spearmanr(x, y)
    sig = '***' if p<.001 else '**' if p<.01 else '*' if p<.05 else 'n.s.'
    return {'test':'Spearman', 'x':x_label, 'y':y_label, 'n':int(len(x)),
            'rho':float(rho), 'p':float(p), 'sig':sig}

def fmt(res):
    if res['test'] == 'MWU':
        return (f"  {res['var']} ({res['label_a']} vs {res['label_b']}): "
                f"{res['label_a']} M={res['M_a']:.3f}±{res['SD_a']:.3f} (n={res['n_a']}), "
                f"{res['label_b']} M={res['M_b']:.3f}±{res['SD_b']:.3f} (n={res['n_b']}), "
                f"U={res['U']:.1f}, p={res['p']:.4f}, r={res['r']:.3f} [{res['sig']}]")
    else:
        return (f"  {res['x']} ↔ {res['y']}: ρ={res['rho']:.3f}, p={res['p']:.4f}, "
                f"n={res['n']} [{res['sig']}]")


def run_additional(output_dir, label):
    with open(os.path.join(output_dir, '_state.pkl'), 'rb') as f:
        state = pickle.load(f)
    df = state['df']
    part_df = state['part_df']

    # Add obj_ratio
    df['obj_ratio'] = df.apply(lambda r: r['obj_count']/(r['ctx_count']+r['obj_count'])
                               if (r['ctx_count']+r['obj_count']) > 0 else np.nan, axis=1)

    # Participant-level keyword aggregates
    part_kw = df.groupby('participant_id').agg(
        mean_ctx_count=('ctx_count','mean'),
        mean_obj_count=('obj_count','mean'),
        mean_ctx_ratio=('ctx_ratio','mean'),
        mean_obj_ratio=('obj_ratio','mean'),
    ).reset_index()
    part_df = part_df.merge(part_kw, on='participant_id', how='left', suffixes=('','_new'))

    fi = df[df['participant_group']=='FI']
    fd = df[df['participant_group']=='FD']
    fi_p = part_df[part_df['participant_group']=='FI']
    fd_p = part_df[part_df['participant_group']=='FD']

    stats = []
    lines = []
    lines.append(f"{'='*65}")
    lines.append(f"  추가 분석 — {label}")
    lines.append(f"{'='*65}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PART A: Axis 1 — 키워드 수 및 비율
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    lines.append("\n## A. Axis 1: 맥락/객체 키워드 수 및 비율 (FI vs FD)\n")

    # Prompt-level
    lines.append("### A-1. 프롬프트 수준 (prompt-level)")
    for var, name in [('ctx_count','맥락키워드수'), ('obj_count','객체키워드수'),
                      ('ctx_ratio','맥락비율'), ('obj_ratio','객체비율')]:
        r = mwu(fi[var], fd[var], name)
        stats.append(r); lines.append(fmt(r))

    # Participant-level
    lines.append("\n### A-2. 참가자 수준 (participant-level)")
    for var, name in [('mean_ctx_count','평균맥락키워드수'), ('mean_obj_count','평균객체키워드수'),
                      ('mean_ctx_ratio_new' if 'mean_ctx_ratio_new' in part_df.columns else 'mean_ctx_ratio','평균맥락비율'),
                      ('mean_obj_ratio','평균객체비율')]:
        if var in part_df.columns:
            r = mwu(fi_p[var], fd_p[var], name)
            stats.append(r); lines.append(fmt(r))

    # Spearman with GEFT (participant-level)
    lines.append("\n### A-3. GEFT 연속 상관 (Spearman)")
    for var, name in [('mean_ctx_count','평균맥락키워드수'), ('mean_obj_count','평균객체키워드수'),
                      ('mean_ctx_ratio_new' if 'mean_ctx_ratio_new' in part_df.columns else 'mean_ctx_ratio','평균맥락비율'),
                      ('mean_obj_ratio','평균객체비율')]:
        if var in part_df.columns:
            r = spear(part_df['geft_score'], part_df[var], 'GEFT', name)
            stats.append(r); lines.append(fmt(r))

    # ── Visualization A ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # A1: ctx_count boxplot
    ax = axes[0,0]
    sns.boxplot(data=df, x='participant_group', y='ctx_count', ax=ax,
                hue='participant_group', palette={'FI':FI_COLOR,'FD':FD_COLOR}, order=['FI','FD'], legend=False)
    sns.stripplot(data=df, x='participant_group', y='ctx_count', ax=ax, color='black', alpha=.3, size=3, order=['FI','FD'])
    ax.set_title('맥락 키워드 수 (FI vs FD)'); ax.set_ylabel('키워드 수')

    # A2: obj_count boxplot
    ax = axes[0,1]
    sns.boxplot(data=df, x='participant_group', y='obj_count', ax=ax,
                hue='participant_group', palette={'FI':FI_COLOR,'FD':FD_COLOR}, order=['FI','FD'], legend=False)
    sns.stripplot(data=df, x='participant_group', y='obj_count', ax=ax, color='black', alpha=.3, size=3, order=['FI','FD'])
    ax.set_title('객체 키워드 수 (FI vs FD)'); ax.set_ylabel('키워드 수')

    # A3: stacked bar (mean ctx vs obj per group)
    ax = axes[0,2]
    fi_ctx_m, fi_obj_m = fi['ctx_count'].mean(), fi['obj_count'].mean()
    fd_ctx_m, fd_obj_m = fd['ctx_count'].mean(), fd['obj_count'].mean()
    x_pos = [0, 1]
    ax.bar(x_pos, [fi_ctx_m, fd_ctx_m], .4, label='맥락', color='#4CAF50', alpha=.8)
    ax.bar(x_pos, [fi_obj_m, fd_obj_m], .4, bottom=[fi_ctx_m, fd_ctx_m], label='객체', color='#FF9800', alpha=.8)
    ax.set_xticks(x_pos); ax.set_xticklabels(['FI','FD'])
    ax.set_title('평균 키워드 구성'); ax.set_ylabel('평균 키워드 수'); ax.legend()

    # A4: ctx_ratio boxplot
    ax = axes[1,0]
    sns.boxplot(data=df.dropna(subset=['ctx_ratio']), x='participant_group', y='ctx_ratio', ax=ax,
                hue='participant_group', palette={'FI':FI_COLOR,'FD':FD_COLOR}, order=['FI','FD'], legend=False)
    ax.set_title('맥락 비율 (FI vs FD)'); ax.set_ylabel('맥락/(맥락+객체)')

    # A5: obj_ratio boxplot
    ax = axes[1,1]
    sns.boxplot(data=df.dropna(subset=['obj_ratio']), x='participant_group', y='obj_ratio', ax=ax,
                hue='participant_group', palette={'FI':FI_COLOR,'FD':FD_COLOR}, order=['FI','FD'], legend=False)
    ax.set_title('객체 비율 (FI vs FD)'); ax.set_ylabel('객체/(맥락+객체)')

    # A6: GEFT scatter with ctx_count & obj_count
    ax = axes[1,2]
    colors = [FI_COLOR if g=='FI' else FD_COLOR for g in part_df['participant_group']]
    ax.scatter(part_df['geft_score'], part_df['mean_ctx_count'], c=colors, marker='o', alpha=.7, s=50, edgecolors='w', label='맥락')
    ax.scatter(part_df['geft_score'], part_df['mean_obj_count'], c=colors, marker='s', alpha=.5, s=50, edgecolors='w', label='객체')
    ax.set_title('GEFT vs 평균 키워드 수'); ax.set_xlabel('GEFT 점수'); ax.set_ylabel('평균 키워드 수'); ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'additional_A_keywords_FIFD.png'), dpi=200, bbox_inches='tight'); plt.close()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PART B: Phase별 맥락/객체 키워드
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    lines.append("\n## B. Phase별 맥락/객체 키워드 차이\n")

    lines.append("### B-1. 전체 (Ideation vs Final)")
    ide = df[df['phase']=='ideation']
    fin = df[df['phase']=='final']
    for var, name in [('ctx_count','맥락키워드수'), ('obj_count','객체키워드수'),
                      ('ctx_ratio','맥락비율'), ('obj_ratio','객체비율')]:
        r = mwu(ide[var], fin[var], name, 'Ideation', 'Final')
        stats.append(r); lines.append(fmt(r))

    lines.append("\n### B-2. FI 그룹 내 (Ideation vs Final)")
    fi_ide = df[(df['participant_group']=='FI') & (df['phase']=='ideation')]
    fi_fin = df[(df['participant_group']=='FI') & (df['phase']=='final')]
    for var, name in [('ctx_count','맥락키워드수'), ('obj_count','객체키워드수'),
                      ('ctx_ratio','맥락비율'), ('obj_ratio','객체비율')]:
        r = mwu(fi_ide[var], fi_fin[var], name, 'FI/Ideation', 'FI/Final')
        stats.append(r); lines.append(fmt(r))

    lines.append("\n### B-3. FD 그룹 내 (Ideation vs Final)")
    fd_ide = df[(df['participant_group']=='FD') & (df['phase']=='ideation')]
    fd_fin = df[(df['participant_group']=='FD') & (df['phase']=='final')]
    for var, name in [('ctx_count','맥락키워드수'), ('obj_count','객체키워드수'),
                      ('ctx_ratio','맥락비율'), ('obj_ratio','객체비율')]:
        r = mwu(fd_ide[var], fd_fin[var], name, 'FD/Ideation', 'FD/Final')
        stats.append(r); lines.append(fmt(r))

    # ── Visualization B ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # B1: Phase × Group ctx_count
    ax = axes[0,0]
    plot_df = df.copy()
    plot_df['group_phase'] = plot_df['participant_group'] + ' / ' + plot_df['phase']
    order = ['FI / ideation','FI / final','FD / ideation','FD / final']
    colors_gp = [FI_COLOR, FI_COLOR, FD_COLOR, FD_COLOR]
    hatches_gp = ['', '///', '', '///']
    means = [plot_df[plot_df['group_phase']==gp]['ctx_count'].mean() for gp in order]
    sds = [plot_df[plot_df['group_phase']==gp]['ctx_count'].std() for gp in order]
    bars = ax.bar(range(4), means, yerr=sds, color=colors_gp, alpha=.7, capsize=4, edgecolor='white')
    for bar, h in zip(bars, hatches_gp):
        bar.set_hatch(h)
    ax.set_xticks(range(4)); ax.set_xticklabels(['FI\nIdeation','FI\nFinal','FD\nIdeation','FD\nFinal'], fontsize=9)
    ax.set_title('맥락 키워드 수 (그룹×Phase)'); ax.set_ylabel('평균 키워드 수')

    # B2: Phase × Group obj_count
    ax = axes[0,1]
    means = [plot_df[plot_df['group_phase']==gp]['obj_count'].mean() for gp in order]
    sds = [plot_df[plot_df['group_phase']==gp]['obj_count'].std() for gp in order]
    bars = ax.bar(range(4), means, yerr=sds, color=colors_gp, alpha=.7, capsize=4, edgecolor='white')
    for bar, h in zip(bars, hatches_gp):
        bar.set_hatch(h)
    ax.set_xticks(range(4)); ax.set_xticklabels(['FI\nIdeation','FI\nFinal','FD\nIdeation','FD\nFinal'], fontsize=9)
    ax.set_title('객체 키워드 수 (그룹×Phase)'); ax.set_ylabel('평균 키워드 수')

    # B3: Phase × Group ctx_ratio
    ax = axes[0,2]
    means = [plot_df[plot_df['group_phase']==gp]['ctx_ratio'].mean() for gp in order]
    sds = [plot_df[plot_df['group_phase']==gp]['ctx_ratio'].std() for gp in order]
    bars = ax.bar(range(4), means, yerr=sds, color=colors_gp, alpha=.7, capsize=4, edgecolor='white')
    for bar, h in zip(bars, hatches_gp):
        bar.set_hatch(h)
    ax.set_xticks(range(4)); ax.set_xticklabels(['FI\nIdeation','FI\nFinal','FD\nIdeation','FD\nFinal'], fontsize=9)
    ax.set_title('맥락 비율 (그룹×Phase)'); ax.set_ylabel('맥락/(맥락+객체)'); ax.set_ylim(0, 1)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PART C: Phase별 발산/수렴
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    lines.append("\n## C. Phase별 발산/수렴 (임베딩 기반)\n")

    # Compute per-prompt cosine distance from participant's first prompt
    from sklearn.metrics.pairwise import cosine_distances
    embeddings = np.load('project_260321/embeddings.npy')
    df_idx = df.reset_index(drop=True)

    # Assign embedding index (prompts are in order)
    prompt_ids = list(df_idx['prompt_id'])

    # Per-participant first-prompt embedding
    first_emb = {}
    for pid, grp in df_idx.groupby('participant_id'):
        first_idx = grp.index[0]
        first_emb[pid] = embeddings[first_idx]

    # Distance from first prompt
    df_idx['dist_from_first'] = [
        float(cosine_distances([embeddings[i]], [first_emb[row['participant_id']]])[0][0])
        for i, row in df_idx.iterrows()
    ]

    # Consecutive distance (already have cos_dist_prev, but let's use it)
    lines.append("### C-1. 전체 (Ideation vs Final)")
    ide_d = df_idx[df_idx['phase']=='ideation']
    fin_d = df_idx[df_idx['phase']=='final']

    for var, name in [('cos_dist_prev','연속거리(턴간)'), ('dist_from_first','첫프롬프트로부터거리')]:
        r = mwu(ide_d[var].dropna(), fin_d[var].dropna(), name, 'Ideation', 'Final')
        stats.append(r); lines.append(fmt(r))

    lines.append("\n### C-2. FI 그룹 내 (Ideation vs Final)")
    fi_ide_d = df_idx[(df_idx['participant_group']=='FI') & (df_idx['phase']=='ideation')]
    fi_fin_d = df_idx[(df_idx['participant_group']=='FI') & (df_idx['phase']=='final')]
    for var, name in [('cos_dist_prev','연속거리'), ('dist_from_first','첫프롬프트거리')]:
        r = mwu(fi_ide_d[var].dropna(), fi_fin_d[var].dropna(), name, 'FI/Ideation', 'FI/Final')
        stats.append(r); lines.append(fmt(r))

    lines.append("\n### C-3. FD 그룹 내 (Ideation vs Final)")
    fd_ide_d = df_idx[(df_idx['participant_group']=='FD') & (df_idx['phase']=='ideation')]
    fd_fin_d = df_idx[(df_idx['participant_group']=='FD') & (df_idx['phase']=='final')]
    for var, name in [('cos_dist_prev','연속거리'), ('dist_from_first','첫프롬프트거리')]:
        r = mwu(fd_ide_d[var].dropna(), fd_fin_d[var].dropna(), name, 'FD/Ideation', 'FD/Final')
        stats.append(r); lines.append(fmt(r))

    # C4: Phase × Group comparison of convergence (Ideation phase FI vs FD, Final phase FI vs FD)
    lines.append("\n### C-4. 같은 Phase 내 FI vs FD")
    for phase_name in ['ideation', 'final']:
        sub_fi = df_idx[(df_idx['participant_group']=='FI') & (df_idx['phase']==phase_name)]
        sub_fd = df_idx[(df_idx['participant_group']=='FD') & (df_idx['phase']==phase_name)]
        for var, name in [('cos_dist_prev',f'{phase_name}_연속거리'),
                          ('dist_from_first',f'{phase_name}_첫프롬프트거리')]:
            r = mwu(sub_fi[var].dropna(), sub_fd[var].dropna(), name)
            stats.append(r); lines.append(fmt(r))

    # ── Visualization C ──
    # B row 2 (axes[1,:]) + C combined
    ax = axes[1,0]
    means = [df_idx[(df_idx['participant_group']==g) & (df_idx['phase']==p)]['cos_dist_prev'].mean()
             for g, p in [('FI','ideation'),('FI','final'),('FD','ideation'),('FD','final')]]
    sds = [df_idx[(df_idx['participant_group']==g) & (df_idx['phase']==p)]['cos_dist_prev'].std()
           for g, p in [('FI','ideation'),('FI','final'),('FD','ideation'),('FD','final')]]
    bars = ax.bar(range(4), means, yerr=sds, color=colors_gp, alpha=.7, capsize=4, edgecolor='white')
    for bar, h in zip(bars, hatches_gp):
        bar.set_hatch(h)
    ax.set_xticks(range(4)); ax.set_xticklabels(['FI\nIdeation','FI\nFinal','FD\nIdeation','FD\nFinal'], fontsize=9)
    ax.set_title('턴간 연속 거리 (그룹×Phase)'); ax.set_ylabel('코사인 거리')

    ax = axes[1,1]
    means = [df_idx[(df_idx['participant_group']==g) & (df_idx['phase']==p)]['dist_from_first'].mean()
             for g, p in [('FI','ideation'),('FI','final'),('FD','ideation'),('FD','final')]]
    sds = [df_idx[(df_idx['participant_group']==g) & (df_idx['phase']==p)]['dist_from_first'].std()
           for g, p in [('FI','ideation'),('FI','final'),('FD','ideation'),('FD','final')]]
    bars = ax.bar(range(4), means, yerr=sds, color=colors_gp, alpha=.7, capsize=4, edgecolor='white')
    for bar, h in zip(bars, hatches_gp):
        bar.set_hatch(h)
    ax.set_xticks(range(4)); ax.set_xticklabels(['FI\nIdeation','FI\nFinal','FD\nIdeation','FD\nFinal'], fontsize=9)
    ax.set_title('첫 프롬프트로부터 거리 (그룹×Phase)'); ax.set_ylabel('코사인 거리')

    # Turn-level trajectory by group & phase
    ax = axes[1,2]
    for g, c in [('FI', FI_COLOR), ('FD', FD_COLOR)]:
        for p, ls, marker in [('ideation', '-', 'o'), ('final', '--', 's')]:
            sub = df_idx[(df_idx['participant_group']==g) & (df_idx['phase']==p)]
            if len(sub) == 0: continue
            turn_agg = sub.groupby('turn_num')['dist_from_first'].mean()
            ax.plot(turn_agg.index, turn_agg.values, color=c, ls=ls, marker=marker, ms=4, alpha=.8,
                    label=f'{g}/{p}')
    ax.set_title('턴별 첫 프롬프트 거리 궤적'); ax.set_xlabel('턴'); ax.set_ylabel('코사인 거리'); ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'additional_BC_phase_keywords_divergence.png'), dpi=200, bbox_inches='tight'); plt.close()

    # Save
    text = '\n'.join(lines)
    print(text)
    with open(os.path.join(output_dir, 'additional_stats.txt'), 'w', encoding='utf-8') as f:
        f.write(text)
    with open(os.path.join(output_dir, 'additional_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\nSaved → {output_dir}/additional_stats.txt, .json, 2 PNGs")
    return stats


if __name__ == '__main__':
    for od, lb in [('output_cut16', 'Cut16 (FI≥16, FD≤15)'), ('output_cut17', 'Cut17 (FI≥16, FD≤15)')]:
        run_additional(od, lb)
    print("\n✓ 추가 분석 완료")
