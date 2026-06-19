import matplotlib.pyplot as plt
import os

def plot_anchors_for_thesis(susp_doc, src_doc, anchors):
    if not anchors:
        print("no anchors for specified pair")
        return

    x_susp = [a[0] for a in anchors]
    y_src = [a[1] for a in anchors]

    plt.figure(figsize=(8, 8))
    plt.scatter(x_susp, y_src, s=10, c='#1f77b4', alpha=0.8, marker='s')

    plt.title(f'lexical line (before dbscan)\n{susp_doc.doc_name} vs {src_doc.doc_name}', pad=20)
    plt.xlabel('suspicious document offset')
    plt.ylabel('source document offset')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

radar_engine = WordEncoplotEngine()

print("loading documents")
doc_susp_5 = Document(5, is_source=False)
doc_src_178 = Document(178, is_source=True)

print(f"extracting anchors between {doc_susp_5.doc_name} and {doc_src_178.doc_name}...")
anchors_5_vs_178 = radar_engine.get_anchors_for_pair(doc_susp_5, doc_src_178)

print(f"number of extracted anchors: {len(anchors_5_vs_178)}")

plot_anchors_for_thesis(doc_susp_5, doc_src_178, anchors_5_vs_178)