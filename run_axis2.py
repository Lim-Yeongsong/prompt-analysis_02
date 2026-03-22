#!/usr/bin/env python3
"""Axis 2: Phase에 따른 구조적 차이 (ideation vs final) — 7 analyses, both cutoffs."""
import os, json
import numpy as np, pandas as pd
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import mannwhitneyu
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from shared import (load_state, setup_font, FI_COLOR, FD_COLOR, IDEATION_COLOR, FINAL_COLOR,
                    mwu_report, spearman_report)

setup_font()
EMB_PATH = 'project_260321/embeddings.npy'

def run_axis2(output_dir, label):
    df, part_df, embeddings, coords, _ = load_state(output_dir, EMB_PATH)
    stats = []
    print(f"\n{'='*60}\n  AXIS 2 — {label}\n{'='*60}")

    # ── 2-1: Phase별 UMAP ──
    print("\n[2-1] Phase별 UMAP 분포")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Phase only
    for ph, c, m in [('ideation', IDEATION_COLOR, 'o'), ('final', FINAL_COLOR, 's')]:
        mask = df['phase'] == ph
        axes[0].scatter(df.loc[mask,'umap_x'], df.loc[mask,'umap_y'], c=c, marker=m, alpha=.6, s=40, label=ph, edgecolors='w', linewidth=.5)
    axes[0].set_title('UMAP: Phase'); axes[0].legend(); axes[0].set_xlabel('UMAP 1'); axes[0].set_ylabel('UMAP 2')

    # Phase x FI/FD
    combos = [('FI','ideation',FI_COLOR,'o'), ('FI','final',FI_COLOR,'s'),
              ('FD','ideation',FD_COLOR,'o'), ('FD','final',FD_COLOR,'s')]
    for g, ph, c, m in combos:
        mask = (df['participant_group']==g) & (df['phase']==ph)
        alpha = .8 if ph=='final' else .4
        axes[1].scatter(df.loc[mask,'umap_x'], df.loc[mask,'umap_y'], c=c, marker=m, alpha=alpha, s=40,
                        label=f'{g}/{ph}', edgecolors='w', linewidth=.5)
    axes[1].set_title('UMAP: Phase × FI/FD'); axes[1].legend(fontsize=8); axes[1].set_xlabel('UMAP 1'); axes[1].set_ylabel('UMAP 2')
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'axis2_1_phase_umap.png'), dpi=200, bbox_inches='tight'); plt.close()

    # ── 2-2: Phase별 맥락/객체 비율 ──
    print("\n[2-2] Phase별 맥락/객체 키워드 비율")
    fig, ax = plt.subplots(figsize=(10, 5))
    df['group_phase'] = df['participant_group'] + ' / ' + df['phase']
    order = ['FI / ideation','FI / final','FD / ideation','FD / final']
    avail = [o for o in order if o in df['group_phase'].values]
    palette_gp = {'FI / ideation':'#64B5F6','FI / final':'#1565C0','FD / ideation':'#FF8A65','FD / final':'#D84315'}
    sns.boxplot(data=df, x='group_phase', y='ctx_ratio', order=avail, ax=ax, palette=palette_gp)
    sns.stripplot(data=df, x='group_phase', y='ctx_ratio', order=avail, ax=ax, color='black', alpha=.3, size=4)
    ax.set_title('맥락 키워드 비율: Phase × FI/FD'); ax.set_ylabel('맥락 비율'); ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=15)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'axis2_2_phase_keywords.png'), dpi=200, bbox_inches='tight'); plt.close()

    for g in ['FI','FD']:
        ide = df[(df['participant_group']==g)&(df['phase']=='ideation')]['ctx_ratio'].dropna()
        fin = df[(df['participant_group']==g)&(df['phase']=='final')]['ctx_ratio'].dropna()
        if len(ide)>0 and len(fin)>0:
            stats.append(mwu_report(ide, fin, f'{g}/ideation', f'{g}/final', 'ctx_ratio'))

    # ── 2-3: Phase별 글자 수 ──
    print("\n[2-3] Phase별 평균 글자 수")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x='group_phase', y='char_count', order=avail, ax=ax, palette=palette_gp)
    sns.stripplot(data=df, x='group_phase', y='char_count', order=avail, ax=ax, color='black', alpha=.3, size=4)
    ax.set_title('글자 수: Phase × FI/FD'); ax.set_ylabel('글자 수'); ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=15)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'axis2_3_phase_charcount.png'), dpi=200, bbox_inches='tight'); plt.close()

    for g in ['FI','FD']:
        ide = df[(df['participant_group']==g)&(df['phase']=='ideation')]['char_count']
        fin = df[(df['participant_group']==g)&(df['phase']=='final')]['char_count']
        stats.append(mwu_report(ide, fin, f'{g}/ideation', f'{g}/final', '글자수'))

    # ── 2-4: Phase별 프롬프트 타입 분포 ──
    print("\n[2-4] Phase별 프롬프트 타입 분포")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, ph in enumerate(['ideation','final']):
        sub = df[df['phase']==ph]
        ct = sub.groupby(['participant_group','prompt_type']).size().unstack(fill_value=0)
        pct = ct.div(ct.sum(axis=1), axis=0) * 100
        pct.plot(kind='bar', stacked=True, ax=axes[i], colormap='Set2', width=.6)
        axes[i].set_title(f'{ph} 프롬프트 타입 비율'); axes[i].set_ylabel('%'); axes[i].tick_params(axis='x', rotation=0)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'axis2_4_phase_prompttype.png'), dpi=200, bbox_inches='tight'); plt.close()

    # ── 2-5: Phase별 이미지 사용 ──
    print("\n[2-5] Phase별 이미지 사용 빈도")
    img_phase = df.groupby(['participant_id','phase','participant_group'])['prompt_img'].apply(lambda x: x.notna().sum()).reset_index(name='img_count')
    fig, ax = plt.subplots(figsize=(10, 5))
    img_phase['group_phase'] = img_phase['participant_group'] + ' / ' + img_phase['phase']
    avail_img = [o for o in order if o in img_phase['group_phase'].values]
    sns.boxplot(data=img_phase, x='group_phase', y='img_count', order=avail_img, ax=ax, palette=palette_gp)
    sns.stripplot(data=img_phase, x='group_phase', y='img_count', order=avail_img, ax=ax, color='black', alpha=.4, size=5)
    ax.set_title('이미지 사용: Phase × FI/FD'); ax.set_ylabel('이미지 수'); ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=15)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'axis2_5_phase_image.png'), dpi=200, bbox_inches='tight'); plt.close()

    # ── 2-6: Phase 전환 지점 ──
    print("\n[2-6] Phase 전환 지점")
    trans = []
    for pid, grp in df.groupby('participant_id'):
        gs = grp.sort_values('turn_num')
        phases = gs['phase'].tolist()
        turns = gs['turn_num'].tolist()
        for i in range(1, len(phases)):
            if phases[i-1]=='ideation' and phases[i]=='final':
                trans.append({'participant_id': pid, 'participant_group': gs.iloc[0]['participant_group'], 'transition_turn': turns[i]})
                break
    trans_df = pd.DataFrame(trans)

    fig, ax = plt.subplots(figsize=(8, 5))
    if len(trans_df) > 0:
        sns.boxplot(data=trans_df, x='participant_group', y='transition_turn', ax=ax,
                    palette={'FI':FI_COLOR,'FD':FD_COLOR}, order=['FI','FD'])
        sns.stripplot(data=trans_df, x='participant_group', y='transition_turn', ax=ax,
                      color='black', alpha=.4, size=5, order=['FI','FD'])
        ax.set_title('Phase 전환 턴 (ideation→final)'); ax.set_ylabel('턴 번호')

        fi_t = trans_df[trans_df['participant_group']=='FI']['transition_turn']
        fd_t = trans_df[trans_df['participant_group']=='FD']['transition_turn']
        if len(fi_t)>0 and len(fd_t)>0:
            stats.append(mwu_report(fi_t.astype(float), fd_t.astype(float), 'FI', 'FD', '전환턴'))
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'axis2_6_phase_transition.png'), dpi=200, bbox_inches='tight'); plt.close()

    # ── 2-7: Phase 전환 거리 ──
    print("\n[2-7] Phase 전환 거리 (마지막 ideation ↔ 첫 final)")
    trans_dist = []
    for pid, grp in df.groupby('participant_id'):
        gs = grp.sort_values('turn_num')
        ide_rows = gs[gs['phase']=='ideation']
        fin_rows = gs[gs['phase']=='final']
        if len(ide_rows)>0 and len(fin_rows)>0:
            last_ide_idx = ide_rows.index[-1]
            first_fin_idx = fin_rows.index[0]
            d = cosine_dist(embeddings[last_ide_idx], embeddings[first_fin_idx])
            trans_dist.append({'participant_id': pid, 'participant_group': gs.iloc[0]['participant_group'], 'transition_distance': d})
    td_df = pd.DataFrame(trans_dist)

    fig, ax = plt.subplots(figsize=(8, 5))
    if len(td_df) > 0:
        sns.boxplot(data=td_df, x='participant_group', y='transition_distance', ax=ax,
                    palette={'FI':FI_COLOR,'FD':FD_COLOR}, order=['FI','FD'])
        sns.stripplot(data=td_df, x='participant_group', y='transition_distance', ax=ax,
                      color='black', alpha=.4, size=5, order=['FI','FD'])
        ax.set_title('Phase 전환 코사인 거리'); ax.set_ylabel('코사인 거리')

        fi_td = td_df[td_df['participant_group']=='FI']['transition_distance']
        fd_td = td_df[td_df['participant_group']=='FD']['transition_distance']
        if len(fi_td)>0 and len(fd_td)>0:
            stats.append(mwu_report(fi_td, fd_td, 'FI', 'FD', '전환거리'))
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'axis2_7_phase_distance.png'), dpi=200, bbox_inches='tight'); plt.close()

    with open(os.path.join(output_dir, '_stats_axis2.json'), 'w') as f:
        json.dump(stats, f, default=str, ensure_ascii=False)
    print(f"\n✓ Axis 2 done → {output_dir}")
    return stats

run_axis2('output_cut16', 'GEFT 16점=FD')
run_axis2('output_cut17', 'GEFT 16점=FI')
