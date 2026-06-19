import os
import glob
import time
import matplotlib.pyplot as plt


def analyze_document_sizes(corpus_path):
    suspicious_dir = os.path.join(corpus_path, "suspicious-document")

    if not os.path.exists(suspicious_dir):
        print(f"[!] Error: suspicious directory not found at {suspicious_dir}")
        return

    short_count = 0
    medium_count = 0
    long_count = 0

    word_counts = []

    txt_pattern = os.path.join(suspicious_dir, "part*", "*.txt")
    txt_files = glob.glob(txt_pattern)
    total_files = len(txt_files)

    print(f"found {total_files} text files to analyze")
    print("starting document size analysis\n")

    start_time = time.time()

    for index, txt_path in enumerate(txt_files, start=1):
        try:
            with open(txt_path, "r", encoding='utf-8', errors='ignore') as f:
                text = f.read()

            word_count = len(text.split())
            word_counts.append(word_count)

            if word_count < 10000:
                short_count += 1
            elif 10000 <= word_count <= 50000:
                medium_count += 1
            else:
                long_count += 1

        except Exception as e:
            print(f"[!] error reading {os.path.basename(txt_path)}: {e}")

        if index % 100 == 0 or index == total_files:
            elapsed = time.time() - start_time
            print(
                f"  [PROGRESS] analyzed {index}/{total_files} text files ({index / total_files * 100:.1f}%) | Elapsed time: {elapsed:.1f}s")

    # plot
    print("\n" + "=" * 50)
    print("Analiza dimensiunilor documentelor")
    print("=" * 50)
    print(f"Documente procesate:    {total_files}")
    print(f"Documente scurte (< 10k cuvinte):    {short_count}")
    print(f"Documente medii (10k-50k cuvinte): {medium_count}")
    print(f"Documente lungi (> 50k cuvinte):     {long_count}")
    print("-" * 50)
    if word_counts:
        print(f"Media numarului de cuvinte:          {int(sum(word_counts) / len(word_counts)):,} words")
        print(f"Numarul maxim de cuvinte:        {max(word_counts):,} words")
        print(f"Numarul minim de cuvinte:        {min(word_counts):,} words")
    print(f"Timp total de executie:        {time.time() - start_time:.1f}s")
    print("=" * 50 + "\n")

    generate_size_plots(short_count, medium_count, long_count, word_counts)


def generate_size_plots(short, medium, long, all_counts):
    plt.figure(figsize=(15, 6))

    # plot 1 - category distribution
    plt.subplot(1, 2, 1)
    categories = ['Short\n(<10k cuvinte)', 'Medium\n(10k-50k cuvinte)', 'Long\n(>50k cuvinte)']
    counts = [short, medium, long]
    colors = ['#7F88D4', '#4F5DD1', '#0515AB']

    plt.bar(categories, counts, color=colors)
    plt.xlabel('Lungimea documentelor')
    plt.ylabel('Numar de documente')
    plt.title('Distributia documentelor in functie de lungime')

    # plot 2 - logarithmic histogram of word counts to show the density shape
    plt.subplot(1, 2, 2)
    plt.hist(all_counts, bins=30, color='#4F5DD1', edgecolor='black', alpha=0.7)
    plt.xlabel('Numar de cuvinte')
    plt.ylabel('Frecventa')
    plt.title('Distributia numarului de cuvinte')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()

    output_chart_path = "./pan11_document_sizes.png"
    plt.savefig(output_chart_path, dpi=300)
    print(f"document size statistics chart saved at: {output_chart_path}")
    plt.show()

if __name__ == "__main__":
    LOCAL_PATH = r"D:\licenta\pan\3250095\pan-plagiarism-corpus-2011.part1\pan-plagiarism-corpus-2011\external-detection-corpus"
    analyze_document_sizes(LOCAL_PATH)