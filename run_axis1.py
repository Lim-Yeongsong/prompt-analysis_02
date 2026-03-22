#!/usr/bin/env python3
"""Axis 1: GEFT 점수에 따른 프롬프트 특성 (7 analyses) — runs for both cut16/cut17."""
import sys, os, re, json
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from shared import (load_state, setup_font, FI_COLOR, FD_COLOR,
                    CONTEXT_KEYWORDS, OBJECT_KEYWORDS, count_keywords,
                    mwu_report, spearman_report, effect_size_r)

setup_font()
EMB_PATH = 'project_260321/embeddings.npy'

def run_axis1(output_dir, label):
    df, part_df, embeddings, coords, _ = load_state(output_dir, EMB_PATH)
    stats = []
    print(f"\n{'='*60}\n  AXIS 1 — {label}\n{'='*60}")

    # ── 1-1: UMAP + Clustering ──
    print("\n[1-1] UMAP + Clustering")
    sil = {k: silhouette_score(embeddings, KMeans(k, random_state=42, n_init=10).fit_predict(embeddings)) for k in range(2, 11)}
    best_k = max(sil, key=sil.get)
    df['cluster'] = KMeans(best_k, random_state=42, n_init=10).fit_predict(embeddings)
    print(f"  Best k={best_k}, silhouette={sil[best_k]:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for g, c, m in [('FI', FI_COLOR, 'o'), ('FD', FD_COLOR, 's')]:
        mask = df['participant_group'] == g
        axes[0].scatter(df.loc[mask, 'umap_x'], df.loc[mask, 'umap_y'], c=c, marker=m, alpha=.7, s=40, label=g, edgecolors='w', linewidth=.5)
    axes[0].set_title('UMAP: FI vs FD'); axes[0].legend(); axes[0].set_xlabel('UMAP 1'); axes[0].set_ylabel('UMAP 2')

    pal = sns.color_palette('Set2', best_k)
    for c_i in range(best_k):
        mask = df['cluster'] == c_i
        axes[1].scatter(df.loc[mask, 'umap_x'], df.loc[mask, 'umap_y'], c=[pal[c_i]], alpha=.7, s=40, label=f'C{c_i}', edgecolors='w', linewidth=.5)
    axes[1].set_title(f'클러스터 (k={best_k})'); axes[1].legend(); axes[1].set_xlabel('UMAP 1'); axes[1].set_ylabel('UMAP 2')

    axes[2].bar(sil.keys(), sil.values(), color='steelblue')
    axes[2].axvline(best_k, color='red', ls='--', label=f'Best k={best_k}')
    axes[2].set_xlabel('k'); axes[2].set_ylabel('Silhouette Score'); axes[2].set_title('실루엣 스코어'); axes[2].legend()
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'axis1_1_umap_clustering.png'), dpi=200, bbox_inches='tight'); plt.close()

    # ── 1-2: Context/Object Keywords ──
    print("\n[1-2] 맥락/객체 키워드")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # FI/FD bar
    kw_agg = df.groupby('participant_group').agg(ctx_m=('ctx_count','mean'), obj_m=('obj_count','mean'),
                                                   ctx_s=('ctx_count','std'), obj_s=('obj_count','std')).reset_index()
    for i, g in enumerate(['FI','FD']):
        r = kw_agg[kw_agg['participant_group']==g].iloc[0]
        axes[0,0].bar(i-.18, r['ctx_m'], .36, yerr=r['ctx_s'], color='#4CAF50', alpha=.8, capsize=3, label='맥락' if i==0 else '')
        axes[0,0].bar(i+.18, r['obj_m'], .36, yerr=r['obj_s'], color='#FF9800', alpha=.8, capsize=3, label='객체' if i==0 else '')
    axes[0,0].set_xticks([0,1]); axes[0,0].set_xticklabels(['FI','FD']); axes[0,0].set_title('FI/FD 평균 키워드 수'); axes[0,0].legend()

    # Turn trend
    tk = df.groupby(['turn_num','participant_group']).agg(ctx_m=('ctx_count','mean'), obj_m=('obj_count','mean')).reset_index()
    for g, c, ls in [('FI',FI_COLOR,'-'),('FD',FD_COLOR,'--')]:
        s = tk[tk['participant_group']==g]
        axes[0,1].plot(s['turn_num'], s['ctx_m'], marker='o', color=c, ls=ls, ms=4, label=f'{g} 맥락')
        axes[0,1].plot(s['turn_num'], s['obj_m'], marker='s', color=c, ls=':', ms=4, label=f'{g} 객체', alpha=.6)
    axes[0,1].set_title('턴별 키워드 추이'); axes[0,1].set_xlabel('턴'); axes[0,1].legend(fontsize=7)

    # Top 15
    for idx2, (g, ax) in enumerate([('FI', axes[1,0]), ('FD', axes[1,1])]):
        sub = df[df['participant_group']==g]
        kw_all = CONTEXT_KEYWORDS + OBJECT_KEYWORDS
        cts = {kw: sum(len(re.findall(re.escape(kw.lower()), str(t).lower())) for t in sub['prompt_combined']) for kw in kw_all}
        cts = {k:v for k,v in cts.items() if v>0}
        top = sorted(cts.items(), key=lambda x:x[1], reverse=True)[:15]
        if top:
            names, vals = zip(*top)
            colors = ['#4CAF50' if n in CONTEXT_KEYWORDS else '#FF9800' for n in names]
            ax.barh(range(len(names)), vals, color=colors, alpha=.8)
            ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9); ax.invert_yaxis()
        ax.set_title(f'{g} Top 15 키워드'); ax.set_xlabel('빈도')
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'axis1_2_keywords.png'), dpi=200, bbox_inches='tight'); plt.close()

    fi_ctx = df[df['participant_group']=='FI']['ctx_ratio'].dropna()
    fd_ctx = df[df['participant_group']=='FD']['ctx_ratio'].dropna()
    stats.append(mwu_report(fi_ctx, fd_ctx, 'FI', 'FD', 'ctx_ratio'))
    stats.append(spearman_report(part_df['geft_score'].values.astype(float), part_df['mean_ctx_ratio'].values.astype(float), 'GEFT', 'ctx_ratio'))

    # ── 1-3: Char Count ──
    print("\n[1-3] 평균 글자 수")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Initial length
    sns.boxplot(data=part_df, x='participant_group', y='initial_prompt_length', ax=axes[0],
                palette={'FI': FI_COLOR, 'FD': FD_COLOR}, order=['FI','FD'])
    sns.stripplot(data=part_df, x='participant_group', y='initial_prompt_length', ax=axes[0],
                  color='black', alpha=.4, size=5, order=['FI','FD'])
    axes[0].set_title('초기 프롬프트 길이'); axes[0].set_ylabel('글자 수')

    # Turn-level char
    tc = df.groupby(['turn_num','participant_group'])['char_count'].mean().reset_index()
    for g, c in [('FI',FI_COLOR),('FD',FD_COLOR)]:
        s = tc[tc['participant_group']==g]
        axes[1].plot(s['turn_num'], s['char_count'], marker='o', color=c, ms=4, label=g)
    axes[1].set_title('턴별 평균 글자 수'); axes[1].set_xlabel('턴'); axes[1].set_ylabel('글자 수'); axes[1].legend()
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'axis1_3_charcount.png'), dpi=200, bbox_inches='tight'); plt.close()

    fi_len = part_df[part_df['participant_group']=='FI']['initial_prompt_length']
    fd_len = part_df[part_df['participant_group']=='FD']['initial_prompt_length']
    stats.append(mwu_report(fi_len, fd_len, 'FI', 'FD', '초기길이'))
    stats.append(spearman_report(part_df['geft_score'].values.astype(float), part_df['initial_prompt_length'].values.astype(float), 'GEFT', '초기길이'))

    # ── 1-4: Prompt Type Distribution ──
    print("\n[1-4] 프롬프트 타입 분포")
    fig, ax = plt.subplots(figsize=(8, 5))
    type_ct = df.groupby(['participant_group','prompt_type']).size().unstack(fill_value=0)
    type_pct = type_ct.div(type_ct.sum(axis=1), axis=0) * 100
    type_pct.plot(kind='bar', stacked=True, ax=ax, colormap='Set2', width=.6)
    ax.set_title('프롬프트 타입 비율 (FI vs FD)'); ax.set_ylabel('%'); ax.set_xlabel(''); ax.tick_params(axis='x', rotation=0)
    ax.legend(title='타입')
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'axis1_4_prompttype.png'), dpi=200, bbox_inches='tight'); plt.close()

    # ── 1-5: Image Usage ──
    print("\n[1-5] 이미지 사용 빈도")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=part_df, x='participant_group', y='img_usage_count', ax=ax,
                palette={'FI': FI_COLOR, 'FD': FD_COLOR}, order=['FI','FD'])
    sns.stripplot(data=part_df, x='participant_group', y='img_usage_count', ax=ax,
                  color='black', alpha=.4, size=5, order=['FI','FD'])
    ax.set_title('참고 이미지 사용 횟수'); ax.set_ylabel('횟수')
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'axis1_5_imageusage.png'), dpi=200, bbox_inches='tight'); plt.close()

    fi_img = part_df[part_df['participant_group']=='FI']['img_usage_count'].astype(float)
    fd_img = part_df[part_df['participant_group']=='FD']['img_usage_count'].astype(float)
    stats.append(mwu_report(fi_img, fd_img, 'FI', 'FD', '이미지사용'))
    stats.append(spearman_report(part_df['geft_score'].values.astype(float), part_df['img_usage_count'].values.astype(float), 'GEFT', '이미지사용'))

    # ── 1-6: Divergence / Convergence ──
    print("\n[1-6] 발산/수렴")
    div_med = part_df['first_last_distance'].median()
    slope_med = part_df['convergence_slope'].median()
    def classify(r):
        hd = r['first_last_distance'] >= div_med
        ns = r['convergence_slope'] <= slope_med
        if hd and not ns: return '지속적 탐색형'
        if hd and ns: return '탐색 후 안착형'
        if not hd and not ns: return '점진적 발산형'
        return '안정적 정교화형'
    part_df['div_type'] = part_df.apply(classify, axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors_bar = [FI_COLOR if g=='FI' else FD_COLOR for g in part_df['participant_group']]
    axes[0].bar(range(len(part_df)), part_df['first_last_distance'], color=colors_bar, alpha=.8)
    axes[0].set_title('발산 (첫-마지막 거리)'); axes[0].set_xlabel('참여자'); axes[0].set_ylabel('코사인 거리')
    axes[0].set_xticks(range(len(part_df))); axes[0].set_xticklabels(part_df['participant_id'], rotation=90, fontsize=6)

    axes[1].bar(range(len(part_df)), part_df['convergence_slope'], color=colors_bar, alpha=.8)
    axes[1].axhline(0, color='black', lw=.5)
    axes[1].set_title('수렴 기울기'); axes[1].set_xlabel('참여자'); axes[1].set_ylabel('기울기')
    axes[1].set_xticks(range(len(part_df))); axes[1].set_xticklabels(part_df['participant_id'], rotation=90, fontsize=6)

    type_c = {'지속적 탐색형':'#E91E63','탐색 후 안착형':'#9C27B0','점진적 발산형':'#FF9800','안정적 정교화형':'#4CAF50'}
    type_m = {'지속적 탐색형':'D','탐색 후 안착형':'o','점진적 발산형':'s','안정적 정교화형':'^'}
    for t in type_c:
        s = part_df[part_df['div_type']==t]
        axes[2].scatter(s['first_last_distance'], s['convergence_slope'], c=type_c[t], marker=type_m[t], s=60, label=t, alpha=.8, edgecolors='w')
    axes[2].axhline(slope_med, color='gray', ls='--', alpha=.5); axes[2].axvline(div_med, color='gray', ls='--', alpha=.5)
    axes[2].set_title('발산 × 수렴 유형'); axes[2].set_xlabel('발산'); axes[2].set_ylabel('수렴 기울기'); axes[2].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'axis1_6_divergence.png'), dpi=200, bbox_inches='tight'); plt.close()

    fi_div = part_df[part_df['participant_group']=='FI']['first_last_distance']
    fd_div = part_df[part_df['participant_group']=='FD']['first_last_distance']
    fi_conv = part_df[part_df['participant_group']=='FI']['convergence_slope']
    fd_conv = part_df[part_df['participant_group']=='FD']['convergence_slope']
    stats.append(mwu_report(fi_div, fd_div, 'FI', 'FD', '발산'))
    stats.append(mwu_report(fi_conv, fd_conv, 'FI', 'FD', '수렴기울기'))
    print("  유형 분포:")
    print(part_df.groupby(['participant_group','div_type']).size().unstack(fill_value=0).to_string())

    # ── 1-7: OSIVQ Correlation ──
    print("\n[1-7] OSIVQ 상관")
    features = ['initial_prompt_length','n_turns','total_change','first_last_distance','img_usage_count','ideation_ratio']
    feat_labels = ['초기길이','턴수','총변화량','첫-마지막거리','이미지사용','아이디에이션비율']
    osivq_dims = ['osivq_object_pct','osivq_spatial_pct','osivq_verbal_pct']
    osivq_labels = ['Object','Spatial','Verbal']

    # GEFT correlations
    print("  GEFT 상관:")
    for f, fl in zip(features, feat_labels):
        stats.append(spearman_report(part_df['geft_score'].values.astype(float), part_df[f].values.astype(float), 'GEFT', fl))
    # OSIVQ correlations
    for od, ol in zip(osivq_dims, osivq_labels):
        print(f"  {ol} 상관:")
        for f, fl in zip(features, feat_labels):
            stats.append(spearman_report(part_df[od].values.astype(float), part_df[f].values.astype(float), ol, fl))

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    from matplotlib.lines import Line2D
    colors_sc = [FI_COLOR if g=='FI' else FD_COLOR for g in part_df['participant_group']]
    for i, (f, fl) in enumerate(zip(features, feat_labels)):
        ax = axes[i//4, i%4]
        ax.scatter(part_df['geft_score'], part_df[f], c=colors_sc, alpha=.7, s=50, edgecolors='w')
        rho, p = spearmanr(part_df['geft_score'], part_df[f])
        ax.set_title(f'GEFT vs {fl}\nρ={rho:.3f}, p={p:.3f}', fontsize=10)
        ax.set_xlabel('GEFT 점수'); ax.set_ylabel(fl)
    # GEFT vs OSIVQ spatial
    ax = axes[1, 2]
    ax.scatter(part_df['osivq_spatial_pct'], part_df['first_last_distance'], c=colors_sc, alpha=.7, s=50, edgecolors='w')
    rho, p = spearmanr(part_df['osivq_spatial_pct'], part_df['first_last_distance'])
    ax.set_title(f'OSIVQ Spatial vs 첫-마지막거리\nρ={rho:.3f}, p={p:.3f}', fontsize=10)
    ax.set_xlabel('OSIVQ Spatial %'); ax.set_ylabel('첫-마지막 거리')
    # GEFT vs OSIVQ spatial
    ax = axes[1, 3]
    ax.scatter(part_df['geft_score'], part_df['osivq_spatial_pct'], c=colors_sc, alpha=.7, s=50, edgecolors='w')
    rho, p = spearmanr(part_df['geft_score'], part_df['osivq_spatial_pct'])
    ax.set_title(f'GEFT vs OSIVQ Spatial\nρ={rho:.3f}, p={p:.3f}', fontsize=10)
    ax.set_xlabel('GEFT 점수'); ax.set_ylabel('OSIVQ Spatial %')
    legend_el = [Line2D([0],[0],marker='o',color='w',markerfacecolor=FI_COLOR,ms=8,label='FI'),
                 Line2D([0],[0],marker='o',color='w',markerfacecolor=FD_COLOR,ms=8,label='FD')]
    fig.legend(handles=legend_el, loc='lower right', fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'axis1_7_osivq.png'), dpi=200, bbox_inches='tight'); plt.close()

    # Save stats
    with open(os.path.join(output_dir, '_stats_axis1.json'), 'w') as f:
        json.dump(stats, f, default=str, ensure_ascii=False)
    print(f"\n✓ Axis 1 done → {output_dir}")
    return stats

s16 = run_axis1('output_cut16', 'GEFT 16점=FD')
s17 = run_axis1('output_cut17', 'GEFT 16점=FI')
