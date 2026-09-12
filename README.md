# Hybrid External Plagiarism Detection System

A hybrid, two-stage system for detecting external plagiarism, including advanced obfuscation such as paraphrasing, syntactic restructuring, and lexical substitution in large collections of text documents. The system combines a fast lexical filtering stage (C++) with a semantic similarity stage (Transformer-based embeddings) to balance computational efficiency with detection accuracy.

## Overview

Modern plagiarism no longer means simple copy-paste. Content is often reworded, restructured, or translated to evade classical string-matching or n-gram-based detectors. This project addresses that gap with a hybrid pipeline:

1. *Lexical analysis* - quickly scans large volumes of text to find exact lexical matches ("anchors"), acting as an initial filter that reduces the search space and eliminates redundant candidate pairs before the more expensive semantic stage.
2. *Semantic analysis* - encodes text fragments into dense vector representations and measures semantic similarity, catching paraphrased or heavily reworded plagiarism that lexical matching alone would miss.

The two components are integrated into a single pipeline, and results can be explored through a simple Gradio interface.


## Lexical Component (C++)

Several approaches were implemented and benchmarked, converging on a final, non-cryptographic hashing strategy for performance:

- Baseline - parallelized **Encoplot** algorithm, using fixed 16-character windows 
- Intermediate - 4-word n-grams, indexed cryptographically with **MD5** 
- **Final** - 4-word n-grams, indexed with the non-cryptographic **FNV-1A** hash 

Operating at a low level with explicit memory management, this component acts as a fast initial filter, discarding redundant candidates and significantly reducing the workload for the semantic stage.

## Semantic Component (Transformers)

Multiple sentence-embedding models were evaluated to identify the best trade-off between inference speed and semantic quality:

- 'all-MiniLM-L6-v2' - compact and fast, good speed/quality trade-off 
- 'paraphrase-multilingual-mpnet-base-v2' - optimized for semantic similarity / paraphrase detection, multilingual 
- 'intfloat/multilingual-e5-large' - designed for semantic search/retrieval; uses '"query:"' / '"passage:"' prefixes per its training methodology 
- 'BAAI/bge-m3' - Multi-Lingual, Multi-Granularity, Multi-Functionality; 100+ languages, handles both short fragments and long documents 

## Clustering

Detected similarity pairs/fragments are grouped using **DBSCAN**, allowing related plagiarism cases to be aggregated without requiring a predefined number of clusters.

## Interface

A lightweight *Gradio* web interface is included for interactively exploring documents and detected plagiarism cases.

## Project Structure

.
├── experiments/          # code used for EDA and diagrams
│   └──  images/          # figures used in the documentation
├── docs/                 # full project documentation
├── licenta.ipynb         # main colab notebook 
└── README.md


## Getting Started

This project was developed and run in **Google Colab**.

1. Open 'licenta.ipynb' in Google Colab.
2. Install dependencies (see the first cells of the notebook).
3. Compile the C++ lexical component.
4. Run the notebook cells in order to reproduce preprocessing, lexical filtering, semantic scoring, and clustering.
5. Launch the Gradio interface to explore results interactively.

## Methodology Notes

Multiple techniques were experimented with at each stage of the pipeline before arriving at the final configuration described above (see the notebook and the full documentation for the comparative experiments and evaluation results).

## Documentation

The full project documentation (methodology details, experimental evaluation, results) is written in Romanian and is available in ['docs/'](./docs). 
