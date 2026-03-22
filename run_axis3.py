#!/usr/bin/env python3
"""Axis 3: 워크숍 그룹 & 전공 (7 analyses) + 종합 요약 — both cutoffs."""
import os, json, datetime
import numpy as np, pandas as pd
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import mannwhitneyu, wilcoxon
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from shared import (load_state, setup_font, FI_COLOR, FD_COLOR,
                    mwu_report, spearman_report)

setup_font()
EMB_PATH = 'project_260321/embeddings.npy'

def run_axis3(output_dir, label):
    df, part_df, embeddings, coords, _ = load_state(output_dir, EMB_PATH)
    stats = []
    print(f"\n{'='*60}\n  AXIS 3 — {label}\n{'='*60}")

    groups = df.groupby('workshop_group')['participant_id'].apply(lambda x: list(x.unique())).to_dict()
    pid_to_grp = df.drop_duplicates('participant_id').set_index('participant_id')['workshop_group'].to_dict()

    # Participant mean embeddings
    part_emb = {}
    for pid, grp in df.groupby('participant_id'):
        part_emb[pid] = embeddings[grp.index.tolist()].mean(axis=0)

    # ── 3-1: 그룹 내 vs 그룹 간 유사도 ──
    print("\n[3-1] 그룹 내 vs 그룹 간 유사도")
    intra, inter = [], []
    for gn, members in groups.items():
        for i in range(len(members)):
            for j in range(i+1, len(members)):
                intra.append(1 - cosine_dist(part_emb[members[i]], part_emb[members[j]]))
    all_p = list(part_emb.keys())
    for i in range(len(all_p)):
        for j in range(i+1, len(all_p)):
            if pid_to_grp[all_p[i]] != pid_to_grp[all_p[j]]:
                inter.append(1 - cosine_dist(part_emb[all_p[i]], part_emb[all_p[j]]))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sim_df = pd.DataFrame([{'similarity': s, 'type': '그룹 내'} for s in intra] + [{'similarity': s, 'type': '그룹 간'} for s in inter])
    sns.boxplot(data=sim_df, x='type', y='similarity', ax=axes[0], palette={'그룹 내':'#4CAF50','그룹 간':'#9E9E9E'})
    axes[0].set_title('그룹 내 vs 그룹 간 유사도'); axes[0].set_ylabel('코사인 유사도')

    stat_intra = np.array(intra); stat_inter = np.array(inter)
    U, p = mannwhitneyu(stat_intra, stat_inter, alternative='greater')
    print(f"  Mann-Whitney U (그룹내 > 그룹간, 단측): U={U:.1f}, p={p:.4f}")
    print(f"    그룹내: n={len(intra)}, M={np.mean(intra):.4f}; 그룹간: n={len(inter)}, M={np.mean(inter):.4f}")
    stats.append({'test':'MWU(one-sided)', 'var':'그룹내vs그룹간', 'U':float(U), 'p':float(p)})

    # 3-1b: UMAP by group
    pal8 = sns.color_palette('Set2', 8)
    for i, gn in enumerate(sorted(groups.keys())):
        mask = df['workshop_group']==gn
        axes[1].scatter(df.loc[mask,'umap_x'], df.loc[mask,'umap_y'], c=[pal8[i]], alpha=.7, s=40, label=gn, edgecolors='w', linewidth=.5)
    axes[1].set_title('UMAP: 워크숍 그룹별'); axes[1].legend(fontsize=7, ncol=2); axes[1].set_xlabel('UMAP 1'); axes[1].set_ylabel('UMAP 2')

    # 3-1c: Group FI/FD composition (axis 3-7)
    comp = df.drop_duplicates('participant_id').groupby(['workshop_group','participant_group']).size().unstack(fill_value=0)
    comp.plot(kind='bar', stacked=True, ax=axes[2], color=[FD_COLOR, FI_COLOR])
    axes[2].set_title('그룹별 FI/FD 구성'); axes[2].set_ylabel('인원수'); axes[2].tick_params(axis='x', rotation=0)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'axis3_1_group_similarity.png'), dpi=200, bbox_inches='tight'); plt.close()

    # ── 3-2: 그룹 토론 효과 ──
    print("\n[3-2] 그룹 토론 효과 (ideation→final 유사도 변화)")
    gsim_records = []
    for gn, members in groups.items():
        for phase in ['ideation','final']:
            pe = {}
            for pid in members:
                pp = df[(df['participant_id']==pid)&(df['phase']==phase)]
                if len(pp)>0:
                    pe[pid] = embeddings[pp.index.tolist()].mean(axis=0)
            sims = []
            pids = list(pe.keys())
            for i in range(len(pids)):
                for j in range(i+1, len(pids)):
                    sims.append(1 - cosine_dist(pe[pids[i]], pe[pids[j]]))
            if sims:
                gsim_records.append({'group':gn, 'phase':phase, 'mean_sim':np.mean(sims)})
    gsim_df = pd.DataFrame(gsim_records)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    if len(gsim_df)>0:
        piv = gsim_df.pivot(index='group', columns='phase', values='mean_sim').dropna()
        if 'ideation' in piv.columns and 'final' in piv.columns:
            x = np.arange(len(piv))
            axes[0].bar(x-.2, piv['ideation'], .4, label='ideation', color='#66BB6A', alpha=.8)
            axes[0].bar(x+.2, piv['final'], .4, label='final', color='#AB47BC', alpha=.8)
            axes[0].set_xticks(x); axes[0].set_xticklabels(piv.index); axes[0].set_title('그룹 내 유사도 변화')
            axes[0].set_ylabel('코사인 유사도'); axes[0].legend()
            if len(piv)>=5:
                W, pw = wilcoxon(piv['final'], piv['ideation'], alternative='greater')
                print(f"  Wilcoxon (final > ideation): W={W:.1f}, p={pw:.4f}")
                stats.append({'test':'Wilcoxon', 'var':'그룹내유사도변화', 'W':float(W), 'p':float(pw)})

    # 3-2b: Centroid distance change
    cent_records = []
    for gn, members in groups.items():
        for phase in ['ideation','final']:
            embs = []
            for pid in members:
                pp = df[(df['participant_id']==pid)&(df['phase']==phase)]
                if len(pp)>0: embs.append(embeddings[pp.index.tolist()].mean(axis=0))
            if len(embs)>=2:
                centroid = np.mean(embs, axis=0)
                dists = [cosine_dist(e, centroid) for e in embs]
                cent_records.append({'group':gn, 'phase':phase, 'mean_dist':np.mean(dists)})
    cent_df = pd.DataFrame(cent_records)
    if len(cent_df)>0:
        cpiv = cent_df.pivot(index='group', columns='phase', values='mean_dist').dropna()
        if 'ideation' in cpiv.columns and 'final' in cpiv.columns:
            x = np.arange(len(cpiv))
            axes[1].bar(x-.2, cpiv['ideation'], .4, label='ideation', color='#66BB6A', alpha=.8)
            axes[1].bar(x+.2, cpiv['final'], .4, label='final', color='#AB47BC', alpha=.8)
            axes[1].set_xticks(x); axes[1].set_xticklabels(cpiv.index); axes[1].set_title('그룹 centroid 거리 변화')
            axes[1].set_ylabel('코사인 거리'); axes[1].legend()

    # ── 3-3: FI/FD별 그룹 토론 영향 ──
    print("\n[3-3] FI/FD별 그룹 토론 영향 차이")
    sc_records = []
    for gn, members in groups.items():
        for pid in members:
            pg = df[df['participant_id']==pid].iloc[0]['participant_group']
            ide = df[(df['participant_id']==pid)&(df['phase']=='ideation')]
            fin = df[(df['participant_id']==pid)&(df['phase']=='final')]
            if len(ide)==0 or len(fin)==0: continue
            e_ide = embeddings[ide.index.tolist()].mean(axis=0)
            e_fin = embeddings[fin.index.tolist()].mean(axis=0)
            s_ide, s_fin = [], []
            for other in members:
                if other==pid: continue
                o_ide = df[(df['participant_id']==other)&(df['phase']=='ideation')]
                o_fin = df[(df['participant_id']==other)&(df['phase']=='final')]
                if len(o_ide)>0: s_ide.append(1-cosine_dist(e_ide, embeddings[o_ide.index.tolist()].mean(axis=0)))
                if len(o_fin)>0: s_fin.append(1-cosine_dist(e_fin, embeddings[o_fin.index.tolist()].mean(axis=0)))
            if s_ide and s_fin:
                sc_records.append({'participant_id':pid,'participant_group':pg,'sim_change':np.mean(s_fin)-np.mean(s_ide)})
    sc_df = pd.DataFrame(sc_records)
    if len(sc_df)>0:
        sns.boxplot(data=sc_df, x='participant_group', y='sim_change', ax=axes[2],
                    palette={'FI':FI_COLOR,'FD':FD_COLOR}, order=['FI','FD'])
        sns.stripplot(data=sc_df, x='participant_group', y='sim_change', ax=axes[2],
                      color='black', alpha=.4, size=5, order=['FI','FD'])
        axes[2].axhline(0, color='gray', ls='--', alpha=.5)
        axes[2].set_title('FI/FD별 유사도 변화폭'); axes[2].set_ylabel('유사도 변화')
        fi_sc = sc_df[sc_df['participant_group']=='FI']['sim_change']
        fd_sc = sc_df[sc_df['participant_group']=='FD']['sim_change']
        if len(fi_sc)>0 and len(fd_sc)>0:
            stats.append(mwu_report(fi_sc, fd_sc, 'FI', 'FD', '유사도변화폭'))
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'axis3_2_group_discussion.png'), dpi=200, bbox_inches='tight'); plt.close()

    # ── 3-4: 전공별 UMAP ──
    print("\n[3-4] 전공별 UMAP")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for major, c, m in [('design','#26A69A','o'),('business','#EF5350','s')]:
        mask = df['major']==major
        axes[0].scatter(df.loc[mask,'umap_x'], df.loc[mask,'umap_y'], c=c, marker=m, alpha=.6, s=40, label=major, edgecolors='w', linewidth=.5)
    axes[0].set_title('UMAP: 전공별'); axes[0].legend(); axes[0].set_xlabel('UMAP 1'); axes[0].set_ylabel('UMAP 2')

    # ── 3-5: 전공 × FI/FD 맥락비율 ──
    print("\n[3-5] 전공 × FI/FD 맥락비율")
    part_df['group_4'] = part_df['major'] + ' / ' + part_df['participant_group']
    order4 = ['design / FI','design / FD','business / FI','business / FD']
    avail4 = [o for o in order4 if o in part_df['group_4'].values]
    sns.boxplot(data=part_df, x='group_4', y='mean_ctx_ratio', order=avail4, ax=axes[1],
                palette=['#64B5F6','#FF8A65','#1565C0','#D84315'])
    sns.stripplot(data=part_df, x='group_4', y='mean_ctx_ratio', order=avail4, ax=axes[1], color='black', alpha=.4, size=5)
    axes[1].set_title('맥락비율: 전공 × FI/FD'); axes[1].set_ylabel('맥락 비율'); axes[1].tick_params(axis='x', rotation=15)

    # ── 3-6: 전공별 초기 길이 ──
    print("\n[3-6] 전공별 초기 프롬프트 길이")
    sns.boxplot(data=part_df, x='major', y='initial_prompt_length', ax=axes[2],
                palette={'design':'#26A69A','business':'#EF5350'}, order=['design','business'])
    sns.stripplot(data=part_df, x='major', y='initial_prompt_length', ax=axes[2], color='black', alpha=.4, size=5, order=['design','business'])
    axes[2].set_title('초기 프롬프트 길이: 전공별'); axes[2].set_ylabel('글자 수')
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'axis3_3_major.png'), dpi=200, bbox_inches='tight'); plt.close()

    d_len = part_df[part_df['major']=='design']['initial_prompt_length']
    b_len = part_df[part_df['major']=='business']['initial_prompt_length']
    stats.append(mwu_report(d_len, b_len, 'design', 'business', '초기길이'))
    d_ctx = part_df[part_df['major']=='design']['mean_ctx_ratio'].dropna()
    b_ctx = part_df[part_df['major']=='business']['mean_ctx_ratio'].dropna()
    if len(d_ctx)>0 and len(b_ctx)>0:
        stats.append(mwu_report(d_ctx, b_ctx, 'design', 'business', '맥락비율'))

    with open(os.path.join(output_dir, '_stats_axis3.json'), 'w') as f:
        json.dump(stats, f, default=str, ensure_ascii=False)
    print(f"\n✓ Axis 3 done → {output_dir}")
    return stats


def generate_summary(output_dir, label):
    """Generate summary markdown for one cutoff."""
    s1 = json.load(open(os.path.join(output_dir, '_stats_axis1.json')))
    s2 = json.load(open(os.path.join(output_dir, '_stats_axis2.json')))
    s3 = json.load(open(os.path.join(output_dir, '_stats_axis3.json')))

    def fmt_stat(s):
        if isinstance(s, dict):
            if s.get('test') == 'Spearman':
                return f"ρ={s.get('rho','N/A')}, p={s.get('p','N/A')}, n={s.get('n','N/A')} [{s.get('sig','?')}]"
            elif 'U' in s:
                return f"U={s.get('U','N/A')}, p={s.get('p','N/A')}, r={s.get('r','N/A')} [{s.get('sig','?')}]"
            elif 'W' in s:
                return f"W={s.get('W','N/A')}, p={s.get('p','N/A')}"
        return str(s)

    lines = [f"# 분석 결과 요약 — {label}", f"**생성**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ""]
    lines.append("## 축 1: GEFT 점수에 따른 프롬프트 특성")
    for s in s1:
        lines.append(f"- {fmt_stat(s)}")
    lines.append("\n## 축 2: Phase에 따른 구조적 차이")
    for s in s2:
        lines.append(f"- {fmt_stat(s)}")
    lines.append("\n## 축 3: 워크숍 그룹 & 전공")
    for s in s3:
        lines.append(f"- {fmt_stat(s)}")

    # File listing
    lines.append("\n## 출력 파일")
    for f in sorted(os.listdir(output_dir)):
        if f.startswith('_'): continue
        sz = os.path.getsize(os.path.join(output_dir, f))
        lines.append(f"- `{f}` ({sz:,} bytes)")

    md = '\n'.join(lines)
    with open(os.path.join(output_dir, 'analysis_summary.md'), 'w', encoding='utf-8') as f:
        f.write(md)
    print(f"Saved: {output_dir}/analysis_summary.md")


def generate_comparison():
    """Compare cut16 vs cut17 results."""
    lines = ["# 두 기준 간 결과 비교 (GEFT 16점=FD vs 16점=FI)", ""]
    lines.append("| 분석 | 변수 | cut16 (16=FD) p값 | cut17 (16=FI) p값 | 일관성 |")
    lines.append("|------|------|-----------------|-----------------|--------|")

    for axis_file in ['_stats_axis1.json', '_stats_axis2.json', '_stats_axis3.json']:
        s16 = json.load(open(os.path.join('output_cut16', axis_file)))
        s17 = json.load(open(os.path.join('output_cut17', axis_file)))
        for i in range(min(len(s16), len(s17))):
            a, b = s16[i], s17[i]
            if isinstance(a, dict) and isinstance(b, dict):
                var = a.get('var', a.get('y', a.get('x', '?')))
                p16 = a.get('p', 'N/A')
                p17 = b.get('p', 'N/A')
                try:
                    p16f, p17f = float(p16), float(p17)
                    same_sig = (p16f < 0.05) == (p17f < 0.05)
                    consistency = "일치" if same_sig else "불일치"
                except:
                    consistency = "?"
                lines.append(f"| {a.get('test','')} | {var} | {p16} | {p17} | {consistency} |")

    md = '\n'.join(lines)
    with open('comparison_summary.md', 'w', encoding='utf-8') as f:
        f.write(md)
    print("Saved: comparison_summary.md")


# Run
run_axis3('output_cut16', 'GEFT 16점=FD')
run_axis3('output_cut17', 'GEFT 16점=FI')

# Summaries
generate_summary('output_cut16', 'GEFT 16점=FD (cut16)')
generate_summary('output_cut17', 'GEFT 16점=FI (cut17)')
generate_comparison()

# Final file listing
print("\n" + "="*60)
for d in ['output_cut16', 'output_cut17']:
    print(f"\n{d}/")
    for f in sorted(os.listdir(d)):
        if f.startswith('_'): continue
        print(f"  {f:45s} {os.path.getsize(os.path.join(d,f)):>10,} bytes")
print(f"\ncomparison_summary.md  {os.path.getsize('comparison_summary.md'):>10,} bytes")
print("\n✓ All analyses complete.")
