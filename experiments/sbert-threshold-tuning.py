import time
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def benchmark_sbert_thresholds(n_susp=50):
    print(f"calibrating sbert threshold (tested on {n_susp} documents)")
    
    global SBERT_THRESHOLD
    original_threshold = SBERT_THRESHOLD
    SBERT_THRESHOLD = 0.40 
    
    global analyzer
    if 'analyzer' not in globals():
        analyzer = SemanticAnalyzer(model_name='paraphrase-multilingual-mpnet-base-v2')
    validator = ValidationMetrics()
    engine_cpp = WordEncoplotEngine(executable="./word_encoplot")
    
    suspicious_docs, source_ids = extract_sample(n_susp)
    source_docs = [Document(s_id, is_source=True) for s_id in source_ids]
    
    print("extracting anchors and computing semantic scores")
    all_raw_fragments = defaultdict(list)
    
    for susp_doc in suspicious_docs:
        src_paths = [d.file_path for d in source_docs]
        raw_results = engine_cpp.run_mass_scan(susp_doc.file_path, src_paths)
        
        pairs = []
        for src_path, anchors in raw_results.items():
            if anchors:
                anchors = engine_cpp.apply_bucketing(anchors, len(susp_doc.text))
                src_doc = next((d for d in source_docs if d.file_path == src_path), None)
                if src_doc:
                    pairs.append((src_doc, anchors))
        
        if pairs:
            verified_fragments, _, _ = analyzer.verify_multiple(susp_doc, pairs)
            all_raw_fragments[susp_doc.numeric_id] = verified_fragments

    thresholds_to_test = np.arange(0.50, 0.95, 0.05)
    
    precisions = []
    recalls = []
    f1_scores = []
    plagdets = []

    print("evaluating metrics for every threshold")
    for t in thresholds_to_test:
        all_results_for_metrics = []
        
        for susp_doc in suspicious_docs:
            doc_cases = []
            valid_fragments = [f for f in all_raw_fragments.get(susp_doc.numeric_id, []) if f.score >= t]
            
            hits_by_source = defaultdict(list)
            for v in valid_fragments:
                hits_by_source[v.src_id].append(v)
                
            for src_id, hits in hits_by_source.items():
                cases = cluster_results_dbscan(hits)
                doc_cases.extend(cases)
                
            m = validator.calculate_metrics(susp_doc.plagiarism_features, doc_cases)
            all_results_for_metrics.append(m)
            
        global_m = validator.calculate_global_score(all_results_for_metrics)
        if not global_m:
            global_m = {'precision': 0, 'recall': 0, 'f1': 0, 'granularity': 1, 'plagdet': 0}
            
        precisions.append(global_m['precision'])
        recalls.append(global_m['recall'])
        f1_scores.append(global_m['f1'])
        plagdets.append(global_m['plagdet'])

    SBERT_THRESHOLD = original_threshold

    # plots
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(thresholds_to_test, precisions, marker='o', linewidth=2, color='#7F88D4', label='Precision')
    ax.plot(thresholds_to_test, recalls, marker='s', linewidth=2, color='#4F5DD1', label='Recall')
    ax.plot(thresholds_to_test, f1_scores, marker='^', linewidth=2, linestyle='--', color='#2537CF', label='F1-Score')
    ax.plot(thresholds_to_test, plagdets, marker='D', linewidth=3, color='#0515AB', label='PlagDet Score')
    
    ax.set_xlabel('Prag similaritate cosinus (SBERT_THRESHOLD)', fontweight='bold')
    ax.set_ylabel('Scor (0.0 - 1.0)', fontweight='bold')
    ax.set_title('Calibrarea pragului de validare semantica', pad=15)
    
    ax.set_xticks(thresholds_to_test)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.legend(loc='lower left', fancybox=True, shadow=True)

    max_plagdet_idx = np.argmax(plagdets)
    optimal_t = thresholds_to_test[max_plagdet_idx]
    optimal_score = plagdets[max_plagdet_idx]
    
    ax.annotate(f'Optim:\nT={optimal_t:.2f}\nPlagDet={optimal_score:.2f}', 
                xy=(optimal_t, optimal_score), 
                xytext=(optimal_t - 0.1, optimal_score + 0.1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

    plt.tight_layout()
    plt.show()

# benchmark_sbert_thresholds(n_susp=50)