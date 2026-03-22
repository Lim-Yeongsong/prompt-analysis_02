"""
Shared preprocessing module for prompt analysis pipeline.
Used by axis1/axis2/axis3 scripts.
"""
import os, re, warnings, pickle
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon, linregress
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

warnings.filterwarnings('ignore')

# ── Font ──
def setup_font():
    cache_dir = matplotlib.get_cachedir()
    for f in os.listdir(cache_dir):
        if f.startswith('fontlist'):
            os.remove(os.path.join(cache_dir, f))
    fm._load_fontmanager(try_read_cache=False)
    # seaborn set_style resets font — call it FIRST, then override
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.1)
    plt.rcParams['font.family'] = 'NanumGothic'
    plt.rcParams['axes.unicode_minus'] = False

setup_font()

# ── Keyword Dictionaries ──
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
FI_COLOR = '#2196F3'
FD_COLOR = '#FF5722'
IDEATION_COLOR = '#66BB6A'
FINAL_COLOR = '#AB47BC'

def count_keywords(text, keyword_list):
    if not isinstance(text, str) or not text.strip():
        return 0
    text_lower = text.lower()
    return sum(len(re.findall(re.escape(kw.lower()), text_lower)) for kw in keyword_list)

def effect_size_r(U, n1, n2):
    return 1 - (2 * U) / (n1 * n2)

def mwu_report(a, b, label_a, label_b, var_name):
    stat, p = mannwhitneyu(a, b, alternative='two-sided')
    r = effect_size_r(stat, len(a), len(b))
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
    txt = (f"  Mann-Whitney U ({var_name} {label_a} vs {label_b}): "
           f"U={stat:.1f}, p={p:.4f}, r={r:.3f} [{sig}]\n"
           f"    {label_a}: n={len(a)}, M={a.mean():.4f}, SD={a.std():.4f}\n"
           f"    {label_b}: n={len(b)}, M={b.mean():.4f}, SD={b.std():.4f}")
    print(txt)
    return {'test': 'Mann-Whitney U', 'var': var_name, 'U': stat, 'p': p, 'r': r,
            f'{label_a}_n': len(a), f'{label_a}_mean': a.mean(),
            f'{label_b}_n': len(b), f'{label_b}_mean': b.mean(), 'sig': sig}

def spearman_report(x, y, x_name, y_name):
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 5:
        print(f"  Spearman ({x_name} vs {y_name}): insufficient data (n={mask.sum()})")
        return {'test': 'Spearman', 'x': x_name, 'y': y_name, 'rho': np.nan, 'p': np.nan, 'n': int(mask.sum())}
    rho, p = spearmanr(x[mask], y[mask])
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
    print(f"  Spearman ({x_name} vs {y_name}): rho={rho:.3f}, p={p:.4f}, n={mask.sum()} [{sig}]")
    return {'test': 'Spearman', 'x': x_name, 'y': y_name, 'rho': rho, 'p': p, 'n': int(mask.sum()), 'sig': sig}


def preprocess(csv_path, embeddings_path, output_dir):
    """Full preprocessing pipeline. Returns (df, part_df, embeddings, coords, stats_log)."""
    os.makedirs(output_dir, exist_ok=True)
    stats_log = []

    # Load
    df = pd.read_csv(csv_path)
    embeddings = np.load(embeddings_path)
    assert embeddings.shape[0] == len(df), f"Embedding count mismatch: {embeddings.shape[0]} vs {len(df)}"
    print(f"Loaded: {csv_path} ({df.shape}), embeddings {embeddings.shape}")

    # Validate columns
    REQUIRED = ['prompt_id','turn','participant_id','major','workshop_group',
                'geft_score','participant_group','osivq_object_pct','osivq_spatial_pct',
                'osivq_verbal_pct','osivq_cognitive_style','phase','prompt_type',
                'prompt_context','prompt_raw','img_type','prompt_img']
    missing = [c for c in REQUIRED if c not in df.columns]
    assert not missing, f"Missing columns: {missing}"

    # Turn number
    df['turn_num'] = df['turn'].str.extract(r'(\d+)').astype(int)

    # Selection replacement
    sel_log = []
    def replace_selection(row):
        if row['prompt_type'] != 'selection':
            return row['prompt_raw']
        raw = str(row['prompt_raw']).strip() if pd.notna(row['prompt_raw']) else ''
        context = str(row['prompt_context']) if pd.notna(row['prompt_context']) else ''
        orig = raw
        if re.match(r'^\d+$', raw):
            num = int(raw)
            for line in context.split('\n'):
                m = re.match(r'[^\d]*(\d+)[^\w]*(.*)', line.strip())
                if m and int(m.group(1)) == num:
                    sel_log.append({'prompt_id': row['prompt_id'], 'original': orig, 'replaced': m.group(2).strip(), 'rule': f'number_{num}'})
                    return m.group(2).strip()
            sel_log.append({'prompt_id': row['prompt_id'], 'original': orig, 'replaced': context, 'rule': 'number_fallback'})
            return context
        pos = r'^(yes|yeah|good|okay|ok|sure|go|엉|응|네|예|좋아|마음에|괜찮|그래|해줘|ㅇㅇ|넹|웅)'
        if re.match(pos, raw, re.IGNORECASE):
            sel_log.append({'prompt_id': row['prompt_id'], 'original': orig, 'replaced': context, 'rule': 'positive'})
            return context if context else raw
        sel_log.append({'prompt_id': row['prompt_id'], 'original': orig, 'replaced': raw, 'rule': 'no_change'})
        return raw
    df['prompt_raw_processed'] = df.apply(replace_selection, axis=1)
    pd.DataFrame(sel_log).to_csv(os.path.join(output_dir, 'selection_log.csv'), index=False, encoding='utf-8-sig')

    # Derived vars
    df['char_count'] = df['prompt_raw_processed'].fillna('').str.len()
    df['char_count_no_space'] = df['prompt_raw_processed'].fillna('').str.replace(' ', '', regex=False).str.len()

    # prompt_combined
    if 'image_caption' in df.columns:
        df['prompt_combined'] = (df['prompt_raw_processed'].fillna('') + ' ' + df['image_caption'].fillna('')).str.strip()
    else:
        df['prompt_combined'] = df['prompt_raw_processed'].fillna('').str.strip()

    # Keywords
    df['ctx_count'] = df['prompt_combined'].apply(lambda x: count_keywords(x, CONTEXT_KEYWORDS))
    df['obj_count'] = df['prompt_combined'].apply(lambda x: count_keywords(x, OBJECT_KEYWORDS))
    df['ctx_ratio'] = df.apply(lambda r: r['ctx_count']/(r['ctx_count']+r['obj_count'])
                               if (r['ctx_count']+r['obj_count']) > 0 else np.nan, axis=1)

    # UMAP
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    coords = reducer.fit_transform(embeddings)
    df['umap_x'] = coords[:, 0]
    df['umap_y'] = coords[:, 1]

    # Cosine distance to previous turn
    df['cos_dist_prev'] = np.nan
    for pid, grp in df.groupby('participant_id'):
        gs = grp.sort_values('turn_num')
        idx = gs.index.tolist()
        for i in range(1, len(idx)):
            df.loc[idx[i], 'cos_dist_prev'] = cosine_dist(embeddings[idx[i-1]], embeddings[idx[i]])

    # Participant-level aggregation
    records = []
    for pid, grp in df.groupby('participant_id'):
        gs = grp.sort_values('turn_num')
        idx = gs.index.tolist()
        emb_p = embeddings[idx]
        initial = gs[gs['prompt_type'] == 'initial']
        init_len = initial['char_count'].values[0] if len(initial) > 0 else gs.iloc[0]['char_count']
        cos_dists = [cosine_dist(emb_p[i-1], emb_p[i]) for i in range(1, len(emb_p))]
        total_change = sum(cos_dists)
        fl_dist = cosine_dist(emb_p[0], emb_p[-1]) if len(emb_p) > 1 else 0
        if len(cos_dists) >= 2:
            slope = linregress(range(len(cos_dists)), cos_dists).slope
        else:
            slope = 0.0
        meta = gs.iloc[0]
        records.append({
            'participant_id': pid, 'participant_group': meta['participant_group'],
            'major': meta['major'], 'workshop_group': meta['workshop_group'],
            'geft_score': meta['geft_score'],
            'osivq_object_pct': meta['osivq_object_pct'],
            'osivq_spatial_pct': meta['osivq_spatial_pct'],
            'osivq_verbal_pct': meta['osivq_verbal_pct'],
            'osivq_cognitive_style': meta['osivq_cognitive_style'],
            'initial_prompt_length': init_len, 'n_turns': len(gs),
            'total_change': total_change, 'first_last_distance': fl_dist,
            'img_usage_count': int(gs['prompt_img'].notna().sum()),
            'ideation_ratio': (gs['phase'] == 'ideation').mean(),
            'mean_char_count': gs['char_count'].mean(),
            'mean_ctx_ratio': gs['ctx_ratio'].mean(),
            'convergence_slope': slope,
        })
    part_df = pd.DataFrame(records)

    # Save
    df.to_csv(os.path.join(output_dir, 'prompt_preprocessed.csv'), index=False, encoding='utf-8-sig')
    part_df.to_csv(os.path.join(output_dir, 'participant_analysis.csv'), index=False, encoding='utf-8-sig')
    np.save(os.path.join(output_dir, 'embeddings.npy'), embeddings)

    # Pickle for inter-script use
    with open(os.path.join(output_dir, '_state.pkl'), 'wb') as f:
        pickle.dump({'df': df, 'part_df': part_df, 'coords': coords, 'stats_log': stats_log}, f)

    print(f"Preprocessing done → {output_dir}")
    return df, part_df, embeddings, coords, stats_log


def load_state(output_dir, embeddings_path):
    with open(os.path.join(output_dir, '_state.pkl'), 'rb') as f:
        state = pickle.load(f)
    embeddings = np.load(embeddings_path)
    return state['df'], state['part_df'], embeddings, state['coords'], state['stats_log']
