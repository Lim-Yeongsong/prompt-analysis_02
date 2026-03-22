#!/usr/bin/env python3
"""
Claude API를 사용한 120개 프롬프트 자동 코딩 (3 프레임워크)
+ FI/FD 분포 차이 분석
"""
import os, json, time, sys
import numpy as np, pandas as pd
import anthropic
from scipy.stats import mannwhitneyu, chi2_contingency, spearmanr
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# Font
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.1)
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

FI_COLOR = '#2196F3'
FD_COLOR = '#FF5722'

SYSTEM_PROMPT = """You are an expert prompt analyst. You will analyze a user's prompt to an AI image generator and code it according to 3 frameworks. The prompt is in Korean.

Return ONLY valid JSON with this exact structure (no markdown, no explanation):
{
  "costar": {
    "context": 0 or 1,
    "objective": 0 or 1,
    "style": 0 or 1,
    "tone": 0 or 1,
    "audience": 0 or 1,
    "response": 0 or 1
  },
  "research_rubrics": {
    "conceptual_scope": "Simple" or "Moderate" or "High",
    "logical_nesting": "Shallow" or "Intermediate" or "Deep",
    "exploratory": "Low" or "Medium" or "High"
  },
  "tian": {
    "task": 0 or 1,
    "persona": 0 or 1,
    "context": 0 or 1,
    "preference": 0 or 1,
    "format": 0 or 1,
    "sample": 0 or 1,
    "regenerate": 0 or 1,
    "visualize": 0 or 1,
    "method": 0 or 1,
    "process": 0 or 1,
    "principle": 0 or 1,
    "constraint": 0 or 1,
    "explain": 0 or 1,
    "reflect": 0 or 1,
    "limitation": 0 or 1
  }
}

## CO-STAR Framework Coding Guide:
- **Context (C)**: Background information or situation description. Does the prompt provide WHY or WHAT SITUATION led to this request? (e.g., "강의실에 콘센트가 부족해서...", "학생들이 충전이 불편해서...")
- **Objective (O)**: Clear goal or desired outcome. Does the prompt state WHAT they want to achieve? (e.g., "이 문제를 해결할 제품을 디자인해줘", "이미지를 만들어줘")
- **Style (S)**: Visual/design style specifications. Does the prompt mention aesthetic style, design language, or visual approach? (e.g., "미니멀한", "모던한", "레트로", "실제 사진처럼")
- **Tone (T)**: Mood or emotional quality. Does the prompt describe atmosphere, feeling, or emotional tone? (e.g., "따뜻한 느낌", "깔끔한", "친근한 분위기")
- **Audience (A)**: Target user/viewer specification. Does the prompt mention WHO will use or see this? (e.g., "학생들을 위한", "대학생 타겟")
- **Response (R)**: Output format specification. Does the prompt specify HOW the response should be formatted? (e.g., "실제 사진처럼", "3D 렌더링으로", "4개의 시안을 보여줘")

## ResearchRubrics Complexity:
- **Conceptual scope**: Simple (1 concept) / Moderate (2-3 concepts) / High (4+ concepts or abstract)
- **Logical nesting**: Shallow (direct request) / Intermediate (conditions/constraints) / Deep (multi-layered logic)
- **Exploratory**: Low (fixed request) / Medium (some exploration) / High (open-ended creative exploration)

## Tian et al. Design Prompt Taxonomy:
### Input
- **Task**: Core design task description (what to create)
- **Persona**: Role/identity assigned to AI
- **Context**: Background/situational information
- **Preference**: Personal preferences or taste
### Output
- **Format**: Output format specification (photo, sketch, 3D, etc.)
- **Sample**: Request for multiple options/samples
- **Regenerate**: Request to modify/redo previous output
- **Visualize**: Request to make visual/render
### Mechanism
- **Method**: Specific technique or method to use
- **Process**: Step-by-step process description
- **Principle**: Design principles or rules to follow
### Control
- **Constraint**: Restrictions or limitations (size, color, material, etc.)
### Blackbox
- **Explain**: Request for explanation
- **Reflect**: Request for evaluation/reflection
- **Limitation**: Acknowledging limitations"""

def code_prompt(client, prompt_text, prompt_id, retry=3):
    """Call Claude API to code a single prompt."""
    for attempt in range(retry):
        try:
            msg = client.messages.create(
                model='claude-sonnet-4-20250514',
                max_tokens=800,
                temperature=0,
                system=SYSTEM_PROMPT,
                messages=[{'role': 'user', 'content': f'Analyze this prompt (ID: {prompt_id}):\n\n{prompt_text}'}]
            )
            text = msg.content[0].text.strip()
            # Clean potential markdown wrapping
            if text.startswith('```'):
                text = text.split('\n', 1)[1] if '\n' in text else text[3:]
                if text.endswith('```'):
                    text = text[:-3]
                text = text.strip()
            result = json.loads(text)
            return result
        except json.JSONDecodeError as e:
            print(f"  JSON parse error on {prompt_id} (attempt {attempt+1}): {e}")
            print(f"  Raw: {text[:200]}")
            if attempt < retry - 1:
                time.sleep(2)
        except anthropic.RateLimitError:
            wait = 2 ** (attempt + 2)
            print(f"  Rate limit on {prompt_id}, waiting {wait}s...")
            time.sleep(wait)
        except Exception as e:
            print(f"  Error on {prompt_id} (attempt {attempt+1}): {e}")
            if attempt < retry - 1:
                time.sleep(2)
    return None


def process_all(csv_path, output_dir):
    """Process all prompts with Claude API."""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    client = anthropic.Anthropic()

    cache_file = os.path.join(output_dir, '_coding_cache.json')
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            cache = json.load(f)
        print(f"Loaded cache: {len(cache)} entries")
    else:
        cache = {}

    total = len(df)
    for i, row in df.iterrows():
        pid = row['prompt_id']
        if pid in cache:
            continue
        prompt_text = str(row['prompt_raw']) if pd.notna(row['prompt_raw']) else ''
        if not prompt_text.strip():
            prompt_text = str(row.get('prompt_context', ''))

        print(f"[{i+1}/{total}] {pid}...", end=' ')
        result = code_prompt(client, prompt_text, pid)
        if result:
            cache[pid] = result
            print("OK")
        else:
            print("FAILED")

        # Save cache periodically
        if (i + 1) % 10 == 0:
            with open(cache_file, 'w') as f:
                json.dump(cache, f, ensure_ascii=False)

        # Rate limiting: ~0.5s between calls
        time.sleep(0.5)

    # Final save
    with open(cache_file, 'w') as f:
        json.dump(cache, f, ensure_ascii=False)
    print(f"\nCoded {len(cache)}/{total} prompts")

    # Build result dataframe
    records = []
    for _, row in df.iterrows():
        pid = row['prompt_id']
        r = {'prompt_id': pid, 'participant_id': row['participant_id'],
             'participant_group': row['participant_group'],
             'geft_score': row['geft_score'], 'phase': row['phase'],
             'prompt_type': row['prompt_type'], 'major': row['major']}

        if pid in cache:
            c = cache[pid]
            # CO-STAR
            cs = c.get('costar', {})
            for k in ['context','objective','style','tone','audience','response']:
                r[f'costar_{k}'] = cs.get(k, 0)
            r['costar_count'] = sum(cs.get(k, 0) for k in ['context','objective','style','tone','audience','response'])

            # ResearchRubrics
            rr = c.get('research_rubrics', {})
            r['rr_conceptual_scope'] = rr.get('conceptual_scope', 'Simple')
            r['rr_logical_nesting'] = rr.get('logical_nesting', 'Shallow')
            r['rr_exploratory'] = rr.get('exploratory', 'Low')

            # Tian
            t = c.get('tian', {})
            for k in ['task','persona','context','preference','format','sample',
                       'regenerate','visualize','method','process','principle',
                       'constraint','explain','reflect','limitation']:
                r[f'tian_{k}'] = t.get(k, 0)
            r['tian_count'] = sum(t.get(k, 0) for k in ['task','persona','context','preference',
                                                          'format','sample','regenerate','visualize',
                                                          'method','process','principle','constraint',
                                                          'explain','reflect','limitation'])
        records.append(r)

    result_df = pd.DataFrame(records)
    result_df.to_csv(os.path.join(output_dir, 'prompt_coded.csv'), index=False, encoding='utf-8-sig')
    print(f"Saved: {output_dir}/prompt_coded.csv")
    return result_df


def analyze_fid(result_df, output_dir):
    """FI/FD distribution analysis with stats."""
    stats_lines = []
    stats_lines.append(f"{'='*60}")
    stats_lines.append(f"  FI/FD 프레임워크 분포 차이 분석 — {output_dir}")
    stats_lines.append(f"{'='*60}\n")

    fi = result_df[result_df['participant_group'] == 'FI']
    fd = result_df[result_df['participant_group'] == 'FD']

    # ── 1. CO-STAR ──
    stats_lines.append("## 1. CO-STAR Framework\n")

    # 1a. costar_count (Mann-Whitney)
    U, p = mannwhitneyu(fi['costar_count'], fd['costar_count'], alternative='two-sided')
    stats_lines.append(f"costar_count: FI M={fi['costar_count'].mean():.2f}±{fi['costar_count'].std():.2f}, "
                       f"FD M={fd['costar_count'].mean():.2f}±{fd['costar_count'].std():.2f}")
    stats_lines.append(f"  Mann-Whitney U={U:.1f}, p={p:.4f} {'*' if p<0.05 else 'n.s.'}\n")

    # Spearman with GEFT
    rho, sp = spearmanr(result_df['geft_score'].astype(float), result_df['costar_count'])
    stats_lines.append(f"  GEFT ↔ costar_count Spearman: ρ={rho:.3f}, p={sp:.4f}\n")

    # 1b. Each element (chi-square)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    costar_elements = ['context','objective','style','tone','audience','response']
    for idx, elem in enumerate(costar_elements):
        col = f'costar_{elem}'
        ct = pd.crosstab(result_df['participant_group'], result_df[col])
        if ct.shape == (2, 2):
            chi2, pv, dof, _ = chi2_contingency(ct)
            sig = '*' if pv < 0.05 else 'n.s.'
            stats_lines.append(f"  costar_{elem}: χ²={chi2:.2f}, p={pv:.4f} [{sig}]")
            stats_lines.append(f"    FI: {fi[col].mean()*100:.1f}%, FD: {fd[col].mean()*100:.1f}%")
        else:
            stats_lines.append(f"  costar_{elem}: insufficient variation for chi-square")

        ax = axes[idx // 4, idx % 4]
        pcts = result_df.groupby('participant_group')[col].mean() * 100
        pcts.plot(kind='bar', ax=ax, color=[FD_COLOR, FI_COLOR], alpha=.8)
        ax.set_title(f'CO-STAR: {elem.title()}'); ax.set_ylabel('%'); ax.tick_params(axis='x', rotation=0)
        ax.set_ylim(0, 100)

    # costar_count box
    ax = axes[1, 2]
    sns.boxplot(data=result_df, x='participant_group', y='costar_count', ax=ax,
                palette={'FI': FI_COLOR, 'FD': FD_COLOR}, order=['FI','FD'])
    sns.stripplot(data=result_df, x='participant_group', y='costar_count', ax=ax,
                  color='black', alpha=.3, size=4, order=['FI','FD'])
    ax.set_title('CO-STAR 포함 요소 수')

    # costar radar
    ax = axes[1, 3]
    angles = np.linspace(0, 2*np.pi, len(costar_elements), endpoint=False).tolist()
    angles += angles[:1]
    fi_vals = [fi[f'costar_{e}'].mean() for e in costar_elements] + [fi[f'costar_{costar_elements[0]}'].mean()]
    fd_vals = [fd[f'costar_{e}'].mean() for e in costar_elements] + [fd[f'costar_{costar_elements[0]}'].mean()]
    ax = fig.add_subplot(2, 4, 8, polar=True)
    ax.plot(angles, fi_vals, 'o-', color=FI_COLOR, label='FI', linewidth=2)
    ax.fill(angles, fi_vals, alpha=.15, color=FI_COLOR)
    ax.plot(angles, fd_vals, 's-', color=FD_COLOR, label='FD', linewidth=2)
    ax.fill(angles, fd_vals, alpha=.15, color=FD_COLOR)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels([e.title() for e in costar_elements], fontsize=8)
    ax.set_title('CO-STAR 레이더', pad=15); ax.legend(fontsize=8, loc='upper right')

    plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'framework_1_costar.png'), dpi=200, bbox_inches='tight'); plt.close()

    # ── 2. ResearchRubrics ──
    stats_lines.append("\n## 2. ResearchRubrics Complexity\n")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    rr_vars = [('rr_conceptual_scope', ['Simple','Moderate','High'], '개념적 범위'),
               ('rr_logical_nesting', ['Shallow','Intermediate','Deep'], '논리적 중첩'),
               ('rr_exploratory', ['Low','Medium','High'], '탐색성')]

    for idx, (col, levels, title) in enumerate(rr_vars):
        ct = pd.crosstab(result_df['participant_group'], result_df[col])
        # Ensure all levels present
        for lv in levels:
            if lv not in ct.columns:
                ct[lv] = 0
        ct = ct[levels]
        if ct.shape[0] >= 2 and ct.values.sum() > 0:
            chi2, pv, dof, _ = chi2_contingency(ct)
            sig = '*' if pv < 0.05 else 'n.s.'
            stats_lines.append(f"  {col}: χ²={chi2:.2f}, p={pv:.4f}, dof={dof} [{sig}]")
        else:
            stats_lines.append(f"  {col}: insufficient data for chi-square")

        pct = ct.div(ct.sum(axis=1), axis=0) * 100
        pct.T.plot(kind='bar', ax=axes[idx], color=[FD_COLOR, FI_COLOR], alpha=.8)
        axes[idx].set_title(f'{title}'); axes[idx].set_ylabel('%'); axes[idx].tick_params(axis='x', rotation=0)
        axes[idx].legend(title='그룹')

    # Numeric encoding for Spearman
    scope_map = {'Simple': 1, 'Moderate': 2, 'High': 3}
    nest_map = {'Shallow': 1, 'Intermediate': 2, 'Deep': 3}
    expl_map = {'Low': 1, 'Medium': 2, 'High': 3}
    result_df['rr_scope_num'] = result_df['rr_conceptual_scope'].map(scope_map)
    result_df['rr_nest_num'] = result_df['rr_logical_nesting'].map(nest_map)
    result_df['rr_expl_num'] = result_df['rr_exploratory'].map(expl_map)

    for col, label in [('rr_scope_num','개념범위'),('rr_nest_num','논리중첩'),('rr_expl_num','탐색성')]:
        rho, sp = spearmanr(result_df['geft_score'].astype(float), result_df[col].astype(float))
        stats_lines.append(f"  GEFT ↔ {label} Spearman: ρ={rho:.3f}, p={sp:.4f}")

    plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'framework_2_rubrics.png'), dpi=200, bbox_inches='tight'); plt.close()

    # ── 3. Tian et al. ──
    stats_lines.append("\n## 3. Tian et al. Design Prompt Taxonomy\n")

    tian_elements = ['task','persona','context','preference','format','sample',
                     'regenerate','visualize','method','process','principle',
                     'constraint','explain','reflect','limitation']
    tian_categories = {
        'Input': ['task','persona','context','preference'],
        'Output': ['format','sample','regenerate','visualize'],
        'Mechanism': ['method','process','principle'],
        'Control': ['constraint'],
        'Blackbox': ['explain','reflect','limitation']
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 3a. Element-level comparison
    ax = axes[0, 0]
    fi_pcts = [fi[f'tian_{e}'].mean()*100 for e in tian_elements]
    fd_pcts = [fd[f'tian_{e}'].mean()*100 for e in tian_elements]
    x = np.arange(len(tian_elements))
    ax.barh(x - 0.2, fi_pcts, 0.4, color=FI_COLOR, alpha=.8, label='FI')
    ax.barh(x + 0.2, fd_pcts, 0.4, color=FD_COLOR, alpha=.8, label='FD')
    ax.set_yticks(x); ax.set_yticklabels(tian_elements, fontsize=9); ax.invert_yaxis()
    ax.set_xlabel('%'); ax.set_title('Tian 요소별 비율'); ax.legend()

    for e in tian_elements:
        col = f'tian_{e}'
        ct = pd.crosstab(result_df['participant_group'], result_df[col])
        if ct.shape == (2, 2) and ct.values.min() > 0:
            chi2, pv, dof, _ = chi2_contingency(ct)
            sig = '*' if pv < 0.05 else 'n.s.'
            stats_lines.append(f"  tian_{e}: FI={fi[col].mean()*100:.1f}%, FD={fd[col].mean()*100:.1f}%, χ²={chi2:.2f}, p={pv:.4f} [{sig}]")
        else:
            stats_lines.append(f"  tian_{e}: FI={fi[col].mean()*100:.1f}%, FD={fd[col].mean()*100:.1f}% (chi-sq N/A)")

    # 3b. Category-level
    ax = axes[0, 1]
    cat_data = []
    for cat, elems in tian_categories.items():
        for g in ['FI','FD']:
            sub = result_df[result_df['participant_group']==g]
            cat_score = sub[[f'tian_{e}' for e in elems]].sum(axis=1).mean()
            cat_data.append({'Category': cat, 'Group': g, 'Mean': cat_score})
    cat_df = pd.DataFrame(cat_data)
    cat_piv = cat_df.pivot(index='Category', columns='Group', values='Mean')
    cat_piv[['FI','FD']].plot(kind='bar', ax=ax, color=[FI_COLOR, FD_COLOR], alpha=.8)
    ax.set_title('Tian 카테고리별 평균 점수'); ax.set_ylabel('평균 요소 수'); ax.tick_params(axis='x', rotation=0)

    # 3c. tian_count
    ax = axes[1, 0]
    sns.boxplot(data=result_df, x='participant_group', y='tian_count', ax=ax,
                palette={'FI': FI_COLOR, 'FD': FD_COLOR}, order=['FI','FD'])
    sns.stripplot(data=result_df, x='participant_group', y='tian_count', ax=ax,
                  color='black', alpha=.3, size=4, order=['FI','FD'])
    ax.set_title('Tian 포함 요소 수')

    U, p = mannwhitneyu(fi['tian_count'], fd['tian_count'], alternative='two-sided')
    stats_lines.append(f"\n  tian_count: FI M={fi['tian_count'].mean():.2f}±{fi['tian_count'].std():.2f}, "
                       f"FD M={fd['tian_count'].mean():.2f}±{fd['tian_count'].std():.2f}")
    stats_lines.append(f"  Mann-Whitney U={U:.1f}, p={p:.4f} {'*' if p<0.05 else 'n.s.'}")
    rho, sp = spearmanr(result_df['geft_score'].astype(float), result_df['tian_count'])
    stats_lines.append(f"  GEFT ↔ tian_count Spearman: ρ={rho:.3f}, p={sp:.4f}")

    # 3d. GEFT scatter
    ax = axes[1, 1]
    colors = [FI_COLOR if g=='FI' else FD_COLOR for g in result_df['participant_group']]
    ax.scatter(result_df['geft_score'], result_df['tian_count'], c=colors, alpha=.6, s=40, edgecolors='w')
    rho2, p2 = spearmanr(result_df['geft_score'].astype(float), result_df['costar_count'])
    ax.set_title(f'GEFT vs Tian Count (ρ={rho:.3f})'); ax.set_xlabel('GEFT 점수'); ax.set_ylabel('Tian 요소 수')

    plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'framework_3_tian.png'), dpi=200, bbox_inches='tight'); plt.close()

    # ── Summary heatmap ──
    fig, ax = plt.subplots(figsize=(14, 6))
    summary_cols = [f'costar_{e}' for e in costar_elements] + [f'tian_{e}' for e in tian_elements]
    summary_labels = [f'CS:{e[:3]}' for e in costar_elements] + [f'T:{e[:4]}' for e in tian_elements]
    hm_data = []
    for g in ['FI','FD']:
        sub = result_df[result_df['participant_group']==g]
        hm_data.append([sub[c].mean()*100 for c in summary_cols])
    hm_df = pd.DataFrame(hm_data, index=['FI','FD'], columns=summary_labels)
    sns.heatmap(hm_df, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax, vmin=0, vmax=100,
                linewidths=.5, cbar_kws={'label': '%'})
    ax.set_title('FI vs FD 프레임워크 요소 비율 (%) 히트맵')
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, 'framework_summary_heatmap.png'), dpi=200, bbox_inches='tight'); plt.close()

    # Save stats
    stats_text = '\n'.join(stats_lines)
    print(stats_text)
    with open(os.path.join(output_dir, 'framework_stats.txt'), 'w', encoding='utf-8') as f:
        f.write(stats_text)
    print(f"\nSaved: {output_dir}/framework_stats.txt")


if __name__ == '__main__':
    CSV = 'project_260321/data/prompt_data_250321_16-FI.csv'
    OUT = 'output_cut17'

    result_df = process_all(CSV, OUT)
    analyze_fid(result_df, OUT)

    # Also for cut16
    CSV16 = 'project_260321/data/prompt_data_250321_16-FD.csv'
    OUT16 = 'output_cut16'
    # Reuse cached coding results (same prompts, different grouping)
    df16 = pd.read_csv(CSV16)
    coded = pd.read_csv(os.path.join(OUT, 'prompt_coded.csv'))
    # Merge coding onto cut16 grouping
    coding_cols = [c for c in coded.columns if c.startswith('costar_') or c.startswith('rr_') or c.startswith('tian_')]
    coded_sub = coded[['prompt_id'] + coding_cols]
    df16_merged = df16.merge(coded_sub, on='prompt_id', how='left')
    df16_merged.to_csv(os.path.join(OUT16, 'prompt_coded.csv'), index=False, encoding='utf-8-sig')
    print(f"\nSaved: {OUT16}/prompt_coded.csv (with cut16 grouping)")
    analyze_fid(df16_merged, OUT16)

    print("\n✓ All framework coding and analysis complete.")
