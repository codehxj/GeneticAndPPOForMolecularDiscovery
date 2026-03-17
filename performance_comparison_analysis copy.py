import os
import json
import math
from typing import List, Dict, Tuple

import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import sys
import pandas as pd
from matplotlib.patches import Patch

from rdkit import Chem

# Project imports
from molecule_generation.molecule_evaluator import MoleculeEvaluator

# DeepDTA predictor
sys.path.append('DeepDTA')
try:
    from affinity_predictor import AffinityPredictor
except Exception:
    AffinityPredictor = None


RESULTS_DIR = os.path.join(os.getcwd(), 'performance_comparison_results')
OUTPUT_DIR = os.path.join(os.getcwd(), 'output')

METHODS = {
    'Base': os.path.join(OUTPUT_DIR, 'base_method', 'molecules.smi'),
    'Enhanced': os.path.join(OUTPUT_DIR, 'enhanced_method', 'molecules.smi'),
    'Intelligent': os.path.join(OUTPUT_DIR, 'intelligent_method', 'molecules.smi'),
}

METHOD_DIRS = {
    'Base': os.path.join(OUTPUT_DIR, 'base_method'),
    'Enhanced': os.path.join(OUTPUT_DIR, 'enhanced_method'),
    'Intelligent': os.path.join(OUTPUT_DIR, 'intelligent_method'),
}

# Color palette for methods
PALETTE = sns.color_palette('Set2', n_colors=3)
METHOD_COLORS = {name: PALETTE[i] for i, name in enumerate(METHODS.keys())}

def method_legend_handles():
    return [Patch(facecolor=METHOD_COLORS[m], edgecolor='k', label=m) for m in METHODS.keys()]


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_smiles(smiles_file: str) -> List[str]:
    smiles = []
    if not os.path.exists(smiles_file):
        return smiles
    with open(smiles_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # allow formats like "SMILES name"
            parts = line.split()
            first = parts[0]
            # skip potential header tokens
            if first.lower() in {"smiles", "#", "smile"}:
                continue
            smiles.append(first)
    return smiles

def load_smiles_from_method_dir(method_dir: str) -> List[str]:
    smiles: List[str] = []
    primary = os.path.join(method_dir, 'molecules.smi')
    smiles += load_smiles(primary)
    for root, _, files in os.walk(method_dir):
        for fn in files:
            if fn.lower().endswith('.smi'):
                fp = os.path.join(root, fn)
                if fp == primary:
                    continue
                smiles += load_smiles(fp)
    return list(dict.fromkeys(smiles))

# anomaly filtering configuration
FILTER_ANOMALIES = True
ANOMALY_RULES = {
    'min_qed': 0.05,
    'max_structural_alerts': 6,
    'max_lipinski_violations': 4,
    'exclude_pains': True
}

MAX_COUNT_PER_METHOD = 2000
MIN_SAMPLES_PER_METHOD = 30

def is_anomalous(record: Dict) -> bool:
    if not FILTER_ANOMALIES:
        return False
    if record.get('QED', 0.0) <= ANOMALY_RULES['min_qed']:
        return True
    if record.get('StructuralAlerts', 0) >= ANOMALY_RULES['max_structural_alerts']:
        return True
    if record.get('LipinskiViolations', 0) > ANOMALY_RULES['max_lipinski_violations']:
        return True
    if ANOMALY_RULES['exclude_pains'] and record.get('IsPAINS', True):
        return True
    return False

# affinity prediction configuration
AFFINITY_CFG = {
    'enabled': True,
    'model_path': 'DeepDTA/deepdta_retrain-prk12-ldk8.pt',
    'ligand_dict_path': 'DeepDTA/ligand_dict-prk12-ldk8.json',
    'protein_dict_path': 'DeepDTA/protein_dict-prk12-ldk8.json',
    'protein_sequence': None,
    'protein_dataset_csv': 'DeepDTA/examples/cleaned_mpro.csv'
}

_affinity_predictor = None
_protein_seq_cache = None

def get_protein_sequence() -> str | None:
    global _protein_seq_cache
    if _protein_seq_cache is not None:
        return _protein_seq_cache
    if isinstance(AFFINITY_CFG.get('protein_sequence'), str) and len(AFFINITY_CFG['protein_sequence']) >= 20:
        _protein_seq_cache = AFFINITY_CFG['protein_sequence']
        return _protein_seq_cache
    try:
        if os.path.exists(AFFINITY_CFG['protein_dataset_csv']):
            df = pd.read_csv(AFFINITY_CFG['protein_dataset_csv'])
            if 'protein' in df.columns:
                seq = str(df.iloc[0]['protein'])
                _protein_seq_cache = seq
                return seq
    except Exception:
        pass
    # fallback example
    _protein_seq_cache = 'MKKFFDSRREQGGSGLGSGSSGFKKSQKDLVAACELGKQSKDLVSQ'
    return _protein_seq_cache

def get_affinity_predictor():
    global _affinity_predictor
    if _affinity_predictor is not None:
        return _affinity_predictor
    if not AFFINITY_CFG['enabled'] or AffinityPredictor is None:
        return None
    try:
        _affinity_predictor = AffinityPredictor(
            model_path=AFFINITY_CFG['model_path'],
            ligand_dict_path=AFFINITY_CFG['ligand_dict_path'],
            protein_dict_path=AFFINITY_CFG['protein_dict_path']
        )
        return _affinity_predictor
    except Exception:
        return None


def compute_metrics_for_method(method_name: str, smiles_list: List[str], max_count: int = MAX_COUNT_PER_METHOD, apply_anomaly_filter: bool = True) -> List[Dict]:
    evaluator = MoleculeEvaluator()
    predictor = get_affinity_predictor()
    protein_seq = get_protein_sequence() if predictor is not None else None
    records = []
    count = 0
    for smi in smiles_list:
        if count >= max_count:
            break
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            fitness, properties, _ = evaluator.calculate_fitness(mol)
            record = {
                'method': method_name,
                'smiles': smi,
                'FitnessScore': fitness,
                'LogP': properties.get('logp', 0.0),
                'TPSA': properties.get('tpsa', 0.0),
                'QED': properties.get('qed', 0.0),
                'MW': properties.get('mw', 0.0),
                'HBA': properties.get('hba', 0.0),
                'HBD': properties.get('hbd', 0.0),
                'RotatableBonds': properties.get('rotatable_bonds', 0.0),
                'HeavyAtoms': properties.get('heavy_atoms', 0.0),
                'NumRings': properties.get('num_rings', 0.0),
                'AromaticAtoms': properties.get('aromatic_atoms', 0.0),
                'Flexibility': properties.get('flexibility', 0.0),
                'TPSA_Density': properties.get('tpsa_density', 0.0),
                'DruglikenessScore': properties.get('druglikeness_score', 0.0),
                'LipinskiViolations': properties.get('lipinski_violations', 0),
                'VeberViolations': properties.get('veber_violations', 0),
                'IsPAINS': properties.get('is_pains', True),
                'StructuralAlerts': properties.get('structural_alerts', 0),
                'IsDruglike': properties.get('is_druglike', False),
                'Affinity': None
            }
            if predictor is not None and protein_seq:
                try:
                    record['Affinity'] = float(predictor.predict_affinity(smi, protein_seq))
                except Exception:
                    record['Affinity'] = None
            if apply_anomaly_filter and is_anomalous(record):
                continue
            records.append(record)
            count += 1
        except Exception:
            continue
    return records


def aggregate_means(records: List[Dict]) -> Dict[str, Dict[str, float]]:
    metrics = [
        'FitnessScore', 'QED', 'DruglikenessScore', 'LogP', 'TPSA', 'MW',
        'RotatableBonds', 'HeavyAtoms', 'NumRings', 'Flexibility',
        'LipinskiViolations', 'VeberViolations', 'StructuralAlerts', 'Affinity'
    ]
    agg: Dict[str, Dict[str, float]] = {}
    by_method: Dict[str, List[Dict]] = {}
    for r in records:
        by_method.setdefault(r['method'], []).append(r)
    # Ensure all methods are present in aggregation, even if some lack records
    for method in METHODS.keys():
        items = by_method.get(method, [])
        agg[method] = {}
        for m in metrics:
            vals = [float(x.get(m, 0.0)) for x in items if m in x]
            agg[method][m] = float(np.mean(vals)) if vals else 0.0
    return agg


def normalize_matrix_for_radar(agg: Dict[str, Dict[str, float]], metrics: List[str]) -> Tuple[Dict[str, List[float]], List[str]]:
    # min-max normalize per metric across methods
    normalized = {}
    mins = {m: math.inf for m in metrics}
    maxs = {m: -math.inf for m in metrics}
    for m in metrics:
        vals = [agg.get(method, {}).get(m, 0.0) for method in agg.keys()]
        if vals:
            mins[m] = float(np.min(vals))
            maxs[m] = float(np.max(vals))
        else:
            mins[m] = 0.0
            maxs[m] = 1.0
    for method in agg.keys():
        normalized[method] = []
        for m in metrics:
            v = agg[method].get(m, 0.0)
            rng = maxs[m] - mins[m]
            nv = 0.0 if rng <= 1e-8 else (v - mins[m]) / rng
            normalized[method].append(nv)
    return normalized, metrics


def save_json(data: Dict, filename: str):
    with open(os.path.join(RESULTS_DIR, filename), 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def plot_counts_bar(stats: Dict[str, Dict[str, int]]):
    methods = list(stats.keys())
    raw = [stats[m].get('records_raw', 0) for m in methods]
    used = [stats[m].get('records_used', 0) for m in methods]
    x = np.arange(len(methods))
    width = 0.35
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, raw, width, label='Raw', color='#8ecae6')
    plt.bar(x + width/2, used, width, label='Used', color='#023047')
    plt.xticks(x, methods)
    plt.ylabel('Record Count')
    plt.title('Record Counts by Method (Raw vs Used)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'bar_counts.png'), dpi=180)
    plt.close()

def plot_affinity_hist(records: List[Dict]):
    methods = list(METHODS.keys())
    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4), sharey=True)
    for i, method in enumerate(methods):
        ax = axes[i] if n > 1 else axes
        vals = [r['Affinity'] for r in records if r['method'] == method and r.get('Affinity') is not None]
        if vals:
            ax.hist(vals, bins=20, color=METHOD_COLORS[method], alpha=0.7)
        ax.set_title(method)
        ax.set_xlabel('Affinity')
        ax.set_ylabel('Count')
    handles = method_legend_handles()
    fig.legend(handles=handles, title='Method', loc='upper right')
    plt.suptitle('Affinity Histogram by Method')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'hist_affinity.png'), dpi=180)
    plt.close()


def plot_hist_mw_tpsa_logp(records: List[Dict]):
    metrics = ['MW', 'TPSA', 'LogP']
    titles = ['MW 分布', 'TPSA 分布', 'LogP 分布']
    plt.figure(figsize=(12, 4))
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = plt.subplot(1, 3, i + 1)
        for method in METHODS.keys():
            vals = [float(r.get(metric, 0.0)) for r in records if r['method'] == method]
            if not vals:
                continue
            ax.hist(vals, bins=30, alpha=0.5, color=METHOD_COLORS[method], label=method)
        ax.set_title(title)
        ax.set_xlabel(metric)
        ax.set_ylabel('Count')
    handles = method_legend_handles()
    plt.legend(handles=handles, title='Method', loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'hist_mw_tpsa_logp.png'), dpi=180)
    plt.close()

def plot_hist_hba_hbd(records: List[Dict]):
    metrics = ['HBA', 'HBD']
    titles = ['HBA 分布', 'HBD 分布']
    plt.figure(figsize=(8, 4))
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = plt.subplot(1, 2, i + 1)
        for method in METHODS.keys():
            vals = [float(r.get(metric, 0.0)) for r in records if r['method'] == method]
            if not vals:
                continue
            ax.hist(vals, bins=30, alpha=0.5, color=METHOD_COLORS[method], label=method)
        ax.set_title(title)
        ax.set_xlabel(metric)
        ax.set_ylabel('Count')
    handles = method_legend_handles()
    plt.legend(handles=handles, title='Method', loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'hist_hba_hbd.png'), dpi=180)
    plt.close()

def plot_box_num_rings(records: List[Dict]):
    plt.figure(figsize=(10, 7))
    sns.set(style='whitegrid')
    data = {
        'Method': [r['method'] for r in records],
        'NumRings': [r.get('NumRings', 0.0) for r in records]
    }
    if not data['Method']:
        plt.close()
        return
    sns.boxplot(x=data['Method'], y=data['NumRings'], palette=[METHOD_COLORS[m] for m in METHODS.keys()])
    plt.xlabel('Method')
    plt.ylabel('Num Rings')
    plt.title('Box Plot: Num Rings by Method')
    handles = method_legend_handles()
    plt.legend(handles=handles, title='Method')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'box_num_rings.png'), dpi=180)
    plt.close()


def plot_bubble(records: List[Dict]):
    # MW vs LogP; bubble size=TPSA; color by method
    plt.figure(figsize=(10, 7))
    sns.set(style='whitegrid')
    for method in METHODS.keys():
        subset = [r for r in records if r['method'] == method]
        if not subset:
            continue
        x = [r['MW'] for r in subset]
        y = [r['LogP'] for r in subset]
        sizes = [max(20, min(800, r['TPSA'])) for r in subset]
        plt.scatter(x, y, s=sizes, alpha=0.5, c=[METHOD_COLORS[method]], label=method, edgecolors='k', linewidths=0.3)
    plt.xlabel('Molecular Weight (MW)')
    plt.ylabel('LogP')
    plt.title('Bubble Chart: MW vs LogP (Bubble size = TPSA)')
    plt.legend(title='Method')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'bubble_mw_logp_tpsa.png'), dpi=180)
    plt.close()


def plot_radar(agg: Dict[str, Dict[str, float]]):
    metrics = ['FitnessScore', 'QED', 'DruglikenessScore', 'Affinity', 'RotatableBonds', 'HeavyAtoms']
    normalized, labels = normalize_matrix_for_radar(agg, metrics)
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(9, 9))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'])
    ax.set_ylim(0, 1)

    for i, (method, values) in enumerate(normalized.items()):
        vals = values + values[:1]
        ax.plot(angles, vals, color=METHOD_COLORS[method], linewidth=2, label=method)
        ax.fill(angles, vals, color=METHOD_COLORS[method], alpha=0.25)

    plt.title('Radar Chart: Normalized Mean Metrics')
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'radar_normalized_means.png'), dpi=180)
    plt.close()


def plot_radial_stacked(agg: Dict[str, Dict[str, float]]):
    metrics = ['QED', 'DruglikenessScore', 'FitnessScore', 'Affinity']
    normalized, labels = normalize_matrix_for_radar(agg, metrics)
    theta = np.linspace(0.0, 2 * np.pi, len(labels), endpoint=False)
    width = 2 * np.pi / len(labels)
    plt.figure(figsize=(9, 9))
    ax = plt.subplot(111, polar=True)
    base = np.zeros_like(theta)
    for method in normalized.keys():
        vals = np.array(normalized.get(method, np.zeros_like(theta)))
        bars = ax.bar(theta, vals, width=width, bottom=base, color=METHOD_COLORS[method], alpha=0.6, edgecolor='white', linewidth=0.8, label=method)
        base += vals
    ax.set_xticks(theta)
    ax.set_xticklabels(labels)
    ax.set_title('Radial Stacked Chart: Normalized Scores')
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'radial_stacked_scores.png'), dpi=180)
    plt.close()


def plot_petal(agg: Dict[str, Dict[str, float]]):
    metrics = ['LogP', 'TPSA', 'MW', 'RotatableBonds', 'HeavyAtoms']
    normalized, labels = normalize_matrix_for_radar(agg, metrics)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    plt.figure(figsize=(9, 9))
    ax = plt.subplot(111, polar=True)
    for method in normalized.keys():
        vals = np.array(normalized.get(method, np.zeros_like(angles)))
        ax.fill(angles, vals, color=METHOD_COLORS[method], alpha=0.25)
        ax.plot(angles, vals, color=METHOD_COLORS[method], linewidth=2, label=method)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    ax.set_title('Petal Chart: Normalized Physicochemical Means')
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'petal_physicochemical_means.png'), dpi=180)
    plt.close()


def plot_fan(agg: Dict[str, Dict[str, float]]):
    metrics = ['LipinskiViolations', 'VeberViolations', 'StructuralAlerts']
    # StructuralAlerts might be zero mean; include if exists
    have_struct = any('StructuralAlerts' in v for v in agg.values())
    if not have_struct:
        metrics = ['LipinskiViolations', 'VeberViolations']
    normalized, labels = normalize_matrix_for_radar(agg, metrics)
    theta = np.linspace(0.0, np.pi, len(labels))  # half circle fan
    width = np.pi / len(labels)
    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    base = 0
    for method in normalized.keys():
        vals = np.array(normalized.get(method, np.zeros_like(theta)))
        ax.bar(theta, vals, width=width, bottom=base, color=METHOD_COLORS[method], alpha=0.6, edgecolor='white', linewidth=0.8, label=method)
    ax.set_xticks(theta)
    ax.set_xticklabels(labels)
    ax.set_title('Fan Chart: Normalized Rule Violations')
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'fan_rule_violations.png'), dpi=180)
    plt.close()


def plot_violin(records: List[Dict]):
    plt.figure(figsize=(10, 7))
    sns.set(style='whitegrid')
    data = {
        'Method': [r['method'] for r in records],
        'QED': [r['QED'] for r in records]
    }
    if not data['Method']:
        plt.close()
        return
    sns.violinplot(x=data['Method'], y=data['QED'], palette=[METHOD_COLORS[m] for m in METHODS.keys()])
    plt.xlabel('Method')
    plt.ylabel('QED')
    plt.title('Violin Plot: QED Distribution by Method')
    handles = method_legend_handles()
    plt.legend(handles=handles, title='Method')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'violin_qed_distribution.png'), dpi=180)
    plt.close()


def plot_box(records: List[Dict]):
    plt.figure(figsize=(10, 7))
    sns.set(style='whitegrid')
    data = {
        'Method': [r['method'] for r in records],
        'DruglikenessScore': [r['DruglikenessScore'] for r in records]
    }
    if not data['Method']:
        plt.close()
        return
    sns.boxplot(x=data['Method'], y=data['DruglikenessScore'], palette=[METHOD_COLORS[m] for m in METHODS.keys()])
    plt.xlabel('Method')
    plt.ylabel('Druglikeness Score')
    plt.title('Box Plot: Druglikeness Score by Method')
    handles = method_legend_handles()
    plt.legend(handles=handles, title='Method')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'box_druglikeness_score.png'), dpi=180)
    plt.close()

def plot_violin_rotatable_bonds(records: List[Dict]):
    plt.figure(figsize=(10, 7))
    sns.set(style='whitegrid')
    data = {
        'Method': [r['method'] for r in records],
        'RotatableBonds': [r.get('RotatableBonds', 0.0) for r in records]
    }
    if not data['Method']:
        plt.close()
        return
    sns.violinplot(x=data['Method'], y=data['RotatableBonds'], palette=[METHOD_COLORS[m] for m in METHODS.keys()])
    plt.xlabel('Method')
    plt.ylabel('Rotatable Bonds')
    plt.title('Violin Plot: Rotatable Bonds Distribution by Method')
    handles = method_legend_handles()
    plt.legend(handles=handles, title='Method')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'violin_rotatable_bonds.png'), dpi=180)
    plt.close()

def plot_box_heavy_atoms(records: List[Dict]):
    plt.figure(figsize=(10, 7))
    sns.set(style='whitegrid')
    data = {
        'Method': [r['method'] for r in records],
        'HeavyAtoms': [r.get('HeavyAtoms', 0.0) for r in records]
    }
    if not data['Method']:
        plt.close()
        return
    sns.boxplot(x=data['Method'], y=data['HeavyAtoms'], palette=[METHOD_COLORS[m] for m in METHODS.keys()])
    plt.xlabel('Method')
    plt.ylabel('Heavy Atoms')
    plt.title('Box Plot: Heavy Atoms by Method')
    handles = method_legend_handles()
    plt.legend(handles=handles, title='Method')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'box_heavy_atoms.png'), dpi=180)
    plt.close()

def plot_affinity_scatter(records: List[Dict]):
    plt.figure(figsize=(10, 7))
    sns.set(style='whitegrid')
    for method in METHODS.keys():
        subset = [r for r in records if r['method'] == method and r.get('Affinity') is not None]
        if not subset:
            continue
        x = [r['DruglikenessScore'] for r in subset]
        y = [r['Affinity'] for r in subset]
        plt.scatter(x, y, alpha=0.6, c=[METHOD_COLORS[method]], label=method, edgecolors='k', linewidths=0.3)
    plt.xlabel('Druglikeness Score')
    plt.ylabel('Predicted Affinity')
    plt.title('Scatter: Affinity vs Druglikeness')
    plt.legend(title='Method')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'scatter_affinity_druglikeness.png'), dpi=180)
    plt.close()

def plot_affinity_violin(records: List[Dict]):
    plt.figure(figsize=(10, 7))
    sns.set(style='whitegrid')
    data = {
        'Method': [r['method'] for r in records if r.get('Affinity') is not None],
        'Affinity': [r['Affinity'] for r in records if r.get('Affinity') is not None]
    }
    if data['Method']:
        sns.violinplot(x=data['Method'], y=data['Affinity'], palette=[METHOD_COLORS[m] for m in METHODS.keys()])
        plt.xlabel('Method')
        plt.ylabel('Predicted Affinity')
        plt.title('Violin Plot: Affinity Distribution by Method')
        handles = method_legend_handles()
        plt.legend(handles=handles, title='Method')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'violin_affinity_distribution.png'), dpi=180)
    plt.close()

def plot_correlation_heatmap(records: List[Dict]):
    cols = ['FitnessScore', 'QED', 'DruglikenessScore', 'LogP', 'TPSA', 'MW', 'RotatableBonds', 'Affinity']
    df = pd.DataFrame([{c: r.get(c) for c in cols} for r in records if r.get('Affinity') is not None])
    if df.empty:
        df = pd.DataFrame([{c: r.get(c) for c in cols} for r in records])
    if df.empty:
        return
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap (Selected Metrics)')
    plt.figtext(0.99, 0.01, '注：热图不区分方法颜色，显示总体相关性', ha='right', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'correlation_heatmap.png'), dpi=180)
    plt.close()

def plot_hexbin_mw_qed(records: List[Dict]):
    methods = list(METHODS.keys())
    fig, axes = plt.subplots(1, len(methods), figsize=(15, 4), sharex=True, sharey=True)
    for i, method in enumerate(methods):
        ax = axes[i]
        subset = [r for r in records if r['method'] == method]
        if subset:
            x = [r['MW'] for r in subset]
            y = [r['QED'] for r in subset]
            hb = ax.hexbin(x, y, gridsize=25, cmap='magma', mincnt=1)
            fig.colorbar(hb, ax=ax)
        ax.set_title(method)
        ax.set_xlabel('MW')
        ax.set_ylabel('QED')
    handles = method_legend_handles()
    fig.legend(handles=handles, title='Method', loc='upper right')
    plt.suptitle('Hexbin Density: MW vs QED by Method')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'hexbin_mw_qed.png'), dpi=180)
    plt.close()

def plot_3d_scatter_mw_logp_affinity(records: List[Dict]):
    methods = list(METHODS.keys())
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    for method in methods:
        subset = [r for r in records if r['method'] == method and r.get('Affinity') is not None]
        if not subset:
            continue
        x = [r['MW'] for r in subset]
        y = [r['LogP'] for r in subset]
        z = [r['Affinity'] for r in subset]
        ax.scatter(x, y, z, color=METHOD_COLORS[method], label=method, alpha=0.8)
    ax.set_xlabel('MW')
    ax.set_ylabel('LogP')
    ax.set_zlabel('Affinity')
    ax.set_title('3D Scatter: MW, LogP, Affinity')
    ax.legend(title='Method')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, '3d_scatter_mw_logp_affinity.png'), dpi=180)
    plt.close()


def plot_3d_waterfall(agg: Dict[str, Dict[str, float]]):
    # 3D bars representing mean values across methods for selected metrics
    metrics = ['FitnessScore', 'QED', 'DruglikenessScore', 'Affinity']
    methods = list(METHODS.keys())
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    xpos, ypos, zpos = [], [], []
    dx, dy, dz = [], [], []
    for i, m in enumerate(methods):
        for j, metric in enumerate(metrics):
            xpos.append(j)
            ypos.append(i)
            zpos.append(0)
            dx.append(0.6)
            dy.append(0.6)
            dz.append(agg.get(m, {}).get(metric, 0.0))
    colors = []
    for i, m in enumerate(methods):
        c = METHOD_COLORS[m]
        for _ in metrics:
            colors.append(c)
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, shade=True)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    handles = method_legend_handles()
    ax.legend(handles=handles, title='Method')
    ax.set_zlabel('Mean Value')
    ax.set_title('3D Waterfall: Mean Metrics Across Methods')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, '3d_waterfall_mean_metrics.png'), dpi=180)
    plt.close()


def plot_3d_stacked_heatmap(agg: Dict[str, Dict[str, float]]):
    # Create a 3D surface per method over property indices
    metrics = ['LogP', 'TPSA', 'MW', 'RotatableBonds', 'HeavyAtoms']
    methods = list(METHODS.keys())
    X = np.arange(len(metrics))
    Y = np.arange(len(methods))
    XX, YY = np.meshgrid(X, Y)
    Z = np.zeros_like(XX, dtype=float)
    for i, m in enumerate(methods):
        for j, metric in enumerate(metrics):
            Z[i, j] = agg.get(m, {}).get(metric, 0.0)
    # Normalize Z for better visualization
    Z = (Z - Z.min()) / (Z.max() - Z.min() + 1e-8)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(XX, YY, Z, cmap='viridis', edgecolor='none', alpha=0.9)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_zlabel('Normalized Value')
    ax.set_title('3D Stacked Planar Heatmap: Normalized Means')
    fig.colorbar(surf, shrink=0.6, aspect=10)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, '3d_stacked_planar_heatmap.png'), dpi=180)
    plt.close()


def build_index_html(image_files: List[str]):
    # 中文结果页：读取统计与均值数据
    stats = {}
    means = {}
    try:
        with open(os.path.join(RESULTS_DIR, 'stats.json'), 'r', encoding='utf-8') as f:
            stats = json.load(f).get('stats', {})
    except Exception:
        pass
    try:
        with open(os.path.join(RESULTS_DIR, 'means.json'), 'r', encoding='utf-8') as f:
            means = json.load(f).get('means', {})
    except Exception:
        pass
    html = [
        '<!doctype html>',
        '<html lang="zh">',
        '<head>',
        '<meta charset="utf-8"/>',
        '<meta name="viewport" content="width=device-width, initial-scale=1"/>',
        '<title>生成方法性能对比</title>',
        '<style>body{font-family:Segoe UI,Arial,sans-serif;background:#f7f7fb;color:#222;padding:20px} .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:20px} .card{background:#fff;border-radius:10px;box-shadow:0 4px 14px rgba(0,0,0,0.07);padding:12px} img{width:100%;border-radius:8px} .section{margin-top:24px} .section h2{margin:8px 0 12px}</style>',
        '</head>',
        '<body>',
        '<h1>生成方法性能对比</h1>',
        '<p>对比 Base、Enhanced、Intelligent 三种分子生成方法。所有图均带统一图例，颜色对应方法。</p>',
        '<h2>汇总统计</h2>',
        '<div class="card">',
        '<table style="width:100%;border-collapse:collapse">',
        '<thead><tr><th style="text-align:left;padding:8px">方法</th><th style="text-align:right;padding:8px">原始数量</th><th style="text-align:right;padding:8px">用于绘图</th><th style="text-align:right;padding:8px">QED</th><th style="text-align:right;padding:8px">Druglikeness</th><th style="text-align:right;padding:8px">Affinity</th></tr></thead>',
        '<tbody>'
    ]
    for method in METHOD_DIRS.keys():
        s = stats.get(method, {})
        m = means.get(method, {})
        html += [
            f'<tr><td style="padding:8px">{method}</td>'
            f'<td style="text-align:right;padding:8px">{s.get("records_raw", 0)}</td>'
            f'<td style="text-align:right;padding:8px">{s.get("records_used", 0)}</td>'
            f'<td style="text-align:right;padding:8px">{m.get("QED", 0.0):.3f}</td>'
            f'<td style="text-align:right;padding:8px">{m.get("DruglikenessScore", 0.0):.3f}</td>'
            f'<td style="text-align:right;padding:8px">{m.get("Affinity", 0.0):.3f}</td></tr>'
        ]
    html += ['</tbody>', '</table>', '</div>']

    # 分组：亲和力相关（包含亲和力分布）
    html += ['<div class="section">', '<h2>亲和力相关分析</h2>', '<p>亲和力与药物相似度的关系、分布以及三维关系。</p>', '<div class="grid">']
    for img in ['hist_affinity.png', 'scatter_affinity_druglikeness.png', 'violin_affinity_distribution.png', '3d_scatter_mw_logp_affinity.png']:
        html += ['<div class="card">', f'<h3>{img.replace("_"," ")}</h3>', f'<img src="{img}" alt="{img}"/>', '</div>']
    html += ['</div>', '</div>']

    # 分组：理化性质分布与关系
    html += ['<div class="section">', '<h2>理化性质分布与关系</h2>', '<p>MW/TPSA/LogP 的分布与相互关系，以及可旋转键与重原子分布。</p>', '<div class="grid">']
    for img in ['bubble_mw_logp_tpsa.png', 'hist_mw_tpsa_logp.png', 'violin_rotatable_bonds.png', 'box_heavy_atoms.png', 'hist_hba_hbd.png']:
        html += ['<div class="card">', f'<h3>{img.replace("_"," ")}</h3>', f'<img src="{img}" alt="{img}"/>', '</div>']
    html += ['</div>', '</div>']

    # 分组：综合指标雷达与堆叠
    html += ['<div class="section">', '<h2>综合指标对比</h2>', '<p>归一化后的多维指标对比，覆盖 QED、Druglikeness、Fitness、Affinity 等。</p>', '<div class="grid">']
    for img in ['radar_normalized_means.png', 'radial_stacked_scores.png', 'petal_physicochemical_means.png', 'fan_rule_violations.png', '3d_waterfall_mean_metrics.png', '3d_stacked_planar_heatmap.png']:
        html += ['<div class="card">', f'<h3>{img.replace("_"," ")}</h3>', f'<img src="{img}" alt="{img}"/>', '</div>']
    html += ['</div>', '</div>']

    # 分组：其他分布与相关性
    html += ['<div class="section">', '<h2>其他分布与相关性</h2>', '<p>QED 与 Druglikeness 的分布，以及 MW 与 QED 的相关性与综合相关性热图。</p>', '<div class="grid">']
    for img in ['violin_qed_distribution.png', 'box_druglikeness_score.png', 'hexbin_mw_qed.png', 'correlation_heatmap.png', 'box_num_rings.png']:
        html += ['<div class="card">', f'<h3>{img.replace("_"," ")}</h3>', f'<img src="{img}" alt="{img}"/>', '</div>']
    html += ['</div>', '</div>']

    html += ['</body>', '</html>']
    with open(os.path.join(RESULTS_DIR, 'index.html'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(html))


def main():
    ensure_dir(RESULTS_DIR)

    # Load SMILES for each method
    method_records_filtered: List[Dict] = []
    method_records_raw: List[Dict] = []
    stats: Dict[str, Dict[str, int]] = {}
    for method, method_dir in METHOD_DIRS.items():
        smiles = load_smiles_from_method_dir(method_dir)
        if not smiles:
            print(f"No SMILES found for method: {method} at {method_dir}")
        raw = compute_metrics_for_method(method, smiles, max_count=MAX_COUNT_PER_METHOD, apply_anomaly_filter=False)
        filtered = compute_metrics_for_method(method, smiles, max_count=MAX_COUNT_PER_METHOD, apply_anomaly_filter=True)
        method_records_raw.extend(raw)
        method_records_filtered.extend(filtered)
        stats[method] = {
            'smiles_total': len(smiles),
            'records_raw': len(raw),
            'records_filtered': len(filtered)
        }

    # Decide records used for plots (fallback if filtered insufficient)
    records_for_plots: List[Dict] = []
    for method in METHOD_DIRS.keys():
        subset_f = [r for r in method_records_filtered if r['method'] == method]
        subset_r = [r for r in method_records_raw if r['method'] == method]
        used = subset_f if len(subset_f) >= MIN_SAMPLES_PER_METHOD else subset_r
        records_for_plots.extend(used)
        stats[method]['records_used'] = len(used)
        stats[method]['used_source'] = 'filtered' if used is subset_f else 'raw'

    # Save records and stats
    save_json({'records_used': records_for_plots, 'records_filtered': method_records_filtered, 'records_raw': method_records_raw, 'stats': stats}, 'records.json')

    # Aggregate means on used records
    agg = aggregate_means(records_for_plots)
    save_json({'means': agg}, 'means.json')
    save_json({'stats': stats}, 'stats.json')

    # Generate plots
    plot_bubble(records_for_plots)
    plot_radar(agg)
    plot_radial_stacked(agg)
    plot_petal(agg)
    plot_fan(agg)
    plot_violin(records_for_plots)
    plot_box(records_for_plots)
    plot_3d_waterfall(agg)
    plot_3d_stacked_heatmap(agg)
    plot_affinity_scatter(records_for_plots)
    plot_affinity_violin(records_for_plots)
    plot_correlation_heatmap(records_for_plots)
    plot_hexbin_mw_qed(records_for_plots)
    plot_3d_scatter_mw_logp_affinity(records_for_plots)
    plot_affinity_hist(records_for_plots)
    plot_hist_mw_tpsa_logp(records_for_plots)
    plot_violin_rotatable_bonds(records_for_plots)
    plot_box_heavy_atoms(records_for_plots)
    plot_hist_hba_hbd(records_for_plots)
    plot_box_num_rings(records_for_plots)

    # Build HTML index
    images = [
        'bubble_mw_logp_tpsa.png',
        'radar_normalized_means.png',
        'radial_stacked_scores.png',
        'petal_physicochemical_means.png',
        'fan_rule_violations.png',
        'violin_qed_distribution.png',
        'violin_rotatable_bonds.png',
        'box_heavy_atoms.png',
        'box_druglikeness_score.png',
        '3d_waterfall_mean_metrics.png',
        '3d_stacked_planar_heatmap.png',
        'scatter_affinity_druglikeness.png',
        'violin_affinity_distribution.png',
        'correlation_heatmap.png',
        'hexbin_mw_qed.png',
        '3d_scatter_mw_logp_affinity.png',
        'hist_affinity.png',
        'hist_mw_tpsa_logp.png',
        'hist_hba_hbd.png',
        'box_num_rings.png',
    ]
    build_index_html(images)
    print(f"Saved results to: {RESULTS_DIR}")


if __name__ == '__main__':
    main()