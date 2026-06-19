import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def benchmark_semantic_models(n_susp=50):
    print(f"benchmarking sbert models ({n_susp} documents)\n")

    suspicious_docs, source_ids = extract_sample(n_susp)
    source_docs = [Document(s_id, is_source=True) for s_id in source_ids]
    
    engine_cpp = WordEncoplotEngine(executable="./word_encoplot")
    validator = ValidationMetrics()
    
    print("[ETAPA 1] Extragerea ancorelor")
    doc_candidate_pairs = {}

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
        doc_candidate_pairs[susp_doc.numeric_id] = pairs
        
    print("anchors have been extracted! starting evaluation\n")

    models_to_test = [
        ('all-MiniLM-L6-v2', 'MiniLM-L6'),
        ('paraphrase-multilingual-mpnet-base-v2', 'MPNet'),
        ('intfloat/multilingual-e5-large', 'E5-Large'),
        ('BAAI/bge-m3', 'BGE-M3')
    ]
    
    results = {}

    for model_id, model_label in models_to_test:
        print(f"loading model: {model_id}")

        torch.cuda.empty_cache()
        
        t_load = time.time()
        analyzer = SemanticAnalyzer(model_name=model_id) 
        print(f"model loaded in {time.time() - t_load:.1f}sec.")
        
        t_sbert_total = 0
        all_metrics_for_model = []
        
        for susp_doc in suspicious_docs:
            pairs = doc_candidate_pairs.get(susp_doc.numeric_id, [])
            doc_cases = []
            
            if pairs:
                t1 = time.time()
                verified_fragments, _, _ = analyzer.verify_multiple(susp_doc, pairs)
                t_sbert_total += (time.time() - t1)

                hits_by_source = defaultdict(list)
                for v in verified_fragments:
                    hits_by_source[v.src_id].append(v)
                for src_id, hits in hits_by_source.items():
                    cases = cluster_results_dbscan(hits)
                    doc_cases.extend(cases)

            m = validator.calculate_metrics(susp_doc.plagiarism_features, doc_cases)
            all_metrics_for_model.append(m)
            
        global_m = validator.calculate_global_score(all_metrics_for_model)
        if not global_m:
            global_m = {'precision': 0, 'recall': 0, 'f1': 0, 'granularity': 1, 'plagdet': 0}
            
        results[model_label] = {
            'time': t_sbert_total,
            'precision': global_m['precision'],
            'recall': global_m['recall'],
            'plagdet': global_m['plagdet']
        }

        del analyzer
        print(f"finalized {model_label}: time = {t_sbert_total:.2f}s | PlagDet = {global_m['plagdet']:.4f}\n")

    # plots
    plt.style.use('ggplot')
    labels = list(results.keys())
    times = [results[l]['time'] for l in labels]
    precisions = [results[l]['precision'] for l in labels]
    recalls = [results[l]['recall'] for l in labels]
    plagdets = [results[l]['plagdet'] for l in labels]
    
    # plot 1 - inference time
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    bars = ax1.bar(labels, times, color='#4F5DD1', width=0.5)
    ax1.set_ylabel('Timp de inferenta (sec)', fontweight='bold')
    ax1.set_title('Timpul total de procesare semantica', pad=15)
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + (max(times)*0.02), f'{yval:.1f}s', ha='center', fontweight='bold', color='#333333')
    plt.tight_layout()
    plt.show()

    # plot 2 - validation metrics
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    x = np.arange(len(labels))
    width = 0.25

    ax2.bar(x - width, precisions, width, label='Precision', color='#7F88D4')
    ax2.bar(x, recalls, width, label='Recall', color='#4F5DD1')
    ax2.bar(x + width, plagdets, width, label='PlagDet Score', color='#0515AB')

    ax2.set_ylabel('Scor (0.0 - 1.0)', fontweight='bold')
    ax2.set_title('Comparatia perfomantei semantice a modelelor', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontweight='bold')
    ax2.set_ylim(0, 1.1)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=3)

    for i in range(len(labels)):
        ax2.text(x[i] - width, precisions[i] + 0.02, f'{precisions[i]:.2f}', ha='center', fontsize=9)
        ax2.text(x[i], recalls[i] + 0.02, f'{recalls[i]:.2f}', ha='center', fontsize=9)
        ax2.text(x[i] + width, plagdets[i] + 0.02, f'{plagdets[i]:.2f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.show()

    # plot 3 - trade off (time vs plagdet)
    fig3, ax3 = plt.subplots(figsize=(9, 6))
    colors = ['#7F88D4', '#4F5DD1', '#2537CF', '#0515AB']
    
    for i, label in enumerate(labels):
        ax3.scatter(times[i], plagdets[i], color=colors[i], s=200, edgecolors='black', label=label.split('\n')[0])
        ax3.annotate(label.split('\n')[0], (times[i], plagdets[i]), xytext=(10, 5), textcoords='offset points', fontweight='bold')

    ax3.set_xlabel('Timp de inferenta (sec)', fontweight='bold')
    ax3.set_ylabel('Scorul PlagDet', fontweight='bold')
    ax3.set_title('Analiza compromisului: Timp versus Performanta', pad=15)
    ax3.grid(True, linestyle='--', alpha=0.7)

    ax3.plot(times, plagdets, linestyle='--', color='gray', alpha=0.5)

    plt.tight_layout()
    plt.show()

# benchmark_semantic_models(n_susp=50)