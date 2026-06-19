import os
import glob
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
import matplotlib.pyplot as plt


def analyze_pan11_corpus(corpus_path):
    suspicious_dir = os.path.join(corpus_path, "suspicious-document")

    if not os.path.exists(suspicious_dir):
        print(f"[!] error: suspicious directory not found at {suspicious_dir}")
        return

    clean_docs_count = 0
    plagiarized_docs_count = 0
    obfuscation_counts = defaultdict(int)
    total_plagiarism_cases = 0

    xml_pattern = os.path.join(suspicious_dir, "part*", "*.xml")
    xml_files = glob.glob(xml_pattern)
    total_files = len(xml_files)

    print(f"found {total_files} XML files to process.")

    start_time = time.time()

    for index, xml_path in enumerate(xml_files, start=1):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            has_plagiarism = False

            for feature in root.findall('feature'):
                if feature.get('name') == 'plagiarism':
                    has_plagiarism = True
                    total_plagiarism_cases += 1

                    obfuscation_type = feature.get('obfuscation', 'none').strip().lower()
                    obfuscation_counts[obfuscation_type] += 1

            if has_plagiarism:
                plagiarized_docs_count += 1
            else:
                clean_docs_count += 1

        except Exception as e:
            print(f"  [!] error parsing {os.path.basename(xml_path)}: {e}")

        # Progress tracking
        if index % 100 == 0 or index == total_files:
            elapsed = time.time() - start_time
            print(
                f"  [PROGRESS] Processed {index}/{total_files} files ({index / total_files * 100:.1f}%) | Elapsed time: {elapsed:.1f}s")

    print("\n" + "=" * 50)
    print("COMPREHENSIVE ANALYSIS REPORT")
    print("=" * 50)
    print(f"processed documents:            {total_files}")
    print(f"original documents:             {clean_docs_count}")
    print(f"plagiarized documents:          {plagiarized_docs_count}")
    print(f"total plagiarism cases:         {total_plagiarism_cases}")
    print("-" * 50)
    print("obfuscation and translation breakdown:")
    for obf_type, count in obfuscation_counts.items():
        marker = " [Cross-Language]" if obf_type == "translation" else ""
        print(f"  - {obf_type}{marker}: {count} cases")
    print(f"Total execution time:        {time.time() - start_time:.1f}sec")
    print("=" * 50 + "\n")

    generate_plots(clean_docs_count, plagiarized_docs_count, obfuscation_counts)


def generate_plots(clean, plagiarized, obfuscation_data):
    plt.figure(figsize=(15, 6))

    # plot 1 - document classification
    plt.subplot(1, 2, 1)
    labels = ['Documente originale', 'Documente plagiate']
    sizes = [clean, plagiarized]
    colors = ['#7F88D4', '#0515AB']
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
    plt.title('Distributia documentelor suspicioase')

    # plot 2 - obfuscation and translation breakdown
    plt.subplot(1, 2, 2)
    types = list(obfuscation_data.keys())
    counts = list(obfuscation_data.values())

    display_types = [t.replace('_', ' ').title() for t in types]

    bar_colors = ['#4242AD' if t == "translation" else '#4F5DD1' for t in types]

    plt.bar(display_types, counts, color=bar_colors)
    plt.xlabel('Obfuscare')
    plt.ylabel('Numar de cazuri')
    plt.title('Analiza cazuri de plagiarism')
    plt.xticks(rotation=30, ha='right')

    plt.tight_layout()

    output_chart_path = "./pan11_corpus_comprehensive_statistics.png"
    plt.savefig(output_chart_path, dpi=300)
    print(f"[OK] Comprehensive statistics chart saved at: {output_chart_path}")
    plt.show()


if __name__ == "__main__":
    LOCAL_PATH = r"D:\licenta\pan\3250095\pan-plagiarism-corpus-2011.part1\pan-plagiarism-corpus-2011\external-detection-corpus"
    analyze_pan11_corpus(LOCAL_PATH)