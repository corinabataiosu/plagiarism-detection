import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from sklearn.cluster import DBSCAN
import copy


def manual_cluster_results(hits, proximity=500):
    if not hits: return []
    hits.sort(key=lambda x: x.susp_off)
    merged = []
    current = copy.copy(hits[0])

    for i in range(1, len(hits)):
        nxt = hits[i]
        if nxt.susp_off <= current.susp_off + current.susp_len + proximity:
            new_end = max(current.susp_off + current.susp_len, nxt.susp_off + nxt.susp_len)
            current.susp_len = new_end - current.susp_off
            current.score = max(current.score, nxt.score)
        else:
            merged.append(current)
            current = copy.copy(nxt)
    merged.append(current)
    return merged


def dbscan_cluster_results(hits, eps_val, min_samples_val):
    if not hits: return []
    if len(hits) < min_samples_val:
        return []

    X = np.array([[h.susp_off, h.src_off] for h in hits])
    clustering = DBSCAN(eps=eps_val, min_samples=min_samples_val).fit(X)
    labels = clustering.labels_

    cases = []
    unique_labels = set(labels)

    for cluster_id in unique_labels:
        if cluster_id == -1:
            continue

        cluster_hits = [hits[i] for i in range(len(hits)) if labels[i] == cluster_id]

        min_susp = min(h.susp_off for h in cluster_hits)
        max_susp = max(h.susp_off + h.susp_len for h in cluster_hits)
        min_src = min(h.src_off for h in cluster_hits)
        max_src = max(h.src_off + h.src_len for h in cluster_hits)

        best_hit = max(cluster_hits, key=lambda h: h.score)

        merged_case = copy.copy(best_hit)
        merged_case.susp_off = min_susp
        merged_case.susp_len = max_susp - min_susp
        merged_case.src_off = min_src
        merged_case.src_len = max_src - min_src

        cases.append(merged_case)

    return cases


def benchmark_clustering_methods(min_susp_id=1, max_susp_id=10):
    print(f"starting clustering algorithm evaluation for documents {min_susp_id} to {max_susp_id}")

    global SBERT_THRESHOLD
    SBERT_THRESHOLD = 0.60

    global analyzer
    if 'analyzer' not in globals():
        analyzer = SemanticAnalyzer(model_name='paraphrase-multilingual-mpnet-base-v2')
    validator = ValidationMetrics()

    # Run pipeline to get raw cached scores instead of extracting them manually
    suspicious_docs, source_docs, _, scores, all_meta = run_pipeline(min_susp_id, max_susp_id, analyzer)

    print("reconstructing verified fragments from pipeline results")
    all_verified_fragments = defaultdict(list)

    for i, score in enumerate(scores):
        if score >= SBERT_THRESHOLD:
            m = all_meta[i]
            hit = PredictedSegment(m['susp_id'], m['src_id'], m['s_off'], m['s_len'], m['src_off'], m['src_len'], score)
            all_verified_fragments[m['susp_id']].append(hit)

    configs = [
        {'name': 'Manual\n(Prox=500)', 'type': 'manual'},
        {'name': 'DBSCAN\n(E=5000, M=3)', 'type': 'dbscan', 'eps': 5000, 'min': 3},
        {'name': 'DBSCAN\n(E=10000, M=3)', 'type': 'dbscan', 'eps': 10000, 'min': 3},
        {'name': 'DBSCAN\n(E=2000, M=2)', 'type': 'dbscan', 'eps': 2000, 'min': 2},
        {'name': 'DBSCAN\n(E=5000, M=2)', 'type': 'dbscan', 'eps': 5000, 'min': 2}
        # Added this config based on user request
    ]

    results = {}
    print("\nevaluating clustering configurations")

    for cfg in configs:
        all_metrics = []
        for susp_doc in suspicious_docs:
            doc_cases = []
            fragments = all_verified_fragments.get(susp_doc.numeric_id, [])

            hits_by_source = defaultdict(list)
            for v in fragments:
                hits_by_source[v.src_id].append(v)

            for src_id, hits in hits_by_source.items():
                if cfg['type'] == 'manual':
                    cases = manual_cluster_results(hits)
                else:
                    cases = dbscan_cluster_results(hits, cfg['eps'], cfg['min'])
                doc_cases.extend(cases)

            m = validator.calculate_metrics(susp_doc.plagiarism_features, doc_cases)
            all_metrics.append(m)

        global_m = validator.calculate_global_score(all_metrics)
        if not global_m:
            global_m = {'precision': 0, 'recall': 0, 'f1': 0, 'granularity': 1, 'plagdet': 0}

        results[cfg['name']] = global_m
        print(f"finalized {cfg['name'].replace(chr(10), ' ')}: PlagDet = {global_m['plagdet']:.4f}")

    # plots
    plt.style.use('ggplot')
    labels = list(results.keys())
    precisions = [results[l]['precision'] for l in labels]
    recalls = [results[l]['recall'] for l in labels]
    granularities = [results[l]['granularity'] for l in labels]
    plagdets = [results[l]['plagdet'] for l in labels]

    x = np.arange(len(labels))
    width = 0.15

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.bar(x - 1.5 * width, precisions, width, label='Precizie', color='#7F88D4')
    ax.bar(x - 0.5 * width, recalls, width, label='Reapel', color='#4F5DD1')
    ax.bar(x + 0.5 * width, granularities, width, label='Granularitate (Ideal=1.0)', color='#2537CF')
    ax.bar(x + 1.5 * width, plagdets, width, label='PlagDet', color='#0515AB')

    ax.set_ylabel('Scor', fontweight='bold')
    ax.set_title('Impactul parametrilor de clusterizare asupra performantei', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight='bold')

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=4)

    for i in range(len(labels)):
        ax.text(x[i] - 1.5 * width, precisions[i] + 0.02, f'{precisions[i]:.2f}', ha='center', fontsize=9)
        ax.text(x[i] - 0.5 * width, recalls[i] + 0.02, f'{recalls[i]:.2f}', ha='center', fontsize=9)
        ax.text(x[i] + 0.5 * width, granularities[i] + 0.02, f'{granularities[i]:.2f}', ha='center', fontsize=9)
        ax.text(x[i] + 1.5 * width, plagdets[i] + 0.02, f'{plagdets[i]:.2f}', ha='center', fontsize=9,
                fontweight='bold')

    plt.tight_layout()
    plt.show()


# Run benchmark on 10 documents by default to limit time, can be increased.
benchmark_clustering_methods(min_susp_id=1, max_susp_id=10)
