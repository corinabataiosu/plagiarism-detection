#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cctype>
#include <stdint.h>

struct Token {
    uint64_t hash;
    int offset;
};

struct NGram {
    uint64_t hash;
    int offset;
};

static uint64_t fnv_word(const std::string& s) {
    uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ULL;
    }
    return h;
}

static uint64_t fnv_combine(uint64_t h, uint64_t x) {
    h ^= x;
    h *= 1099511628211ULL;
    return h;
}

std::vector<Token> tokenize(const std::string& text) {
    std::vector<Token> result;
    std::string word;
    int start = -1;

    for (int i = 0; i < (int)text.size(); i++) {
        char c = text[i];

        if (c == ' ' || c == '\n' || c == '\t' || c == '\r' ||
            c == '.' || c == ','  || c == '!'  || c == '?'  ||
            c == ';' || c == ':'  || c == '('  || c == ')') {

            if (!word.empty()) {
                result.push_back({
                    fnv_word(word),
                    start
                });
                word.clear();
            }
            start = -1;
        }
        else {
            if (word.empty())
                start = i;
            if (c >= 'A' && c <= 'Z') {
                c = c + 32;
            }
            word.push_back(c);
        }
    }

    if (!word.empty()) {
        result.push_back({
            fnv_word(word),
            start
        });
    }

    return result;
}

std::vector<NGram> build_ngrams(const std::vector<Token>& tokens, int n) {
    std::vector<NGram> result;
    if ((int)tokens.size() < n)
        return result;

    for (int i = 0; i <= (int)tokens.size() - n; i++) {
        uint64_t h = 1469598103934665603ULL;
        for (int j = 0; j < n; j++) {
            h = fnv_combine(h, tokens[i+j].hash);
        }
        result.push_back({
            h,
            tokens[i].offset
        });
    }
    return result;
}

std::string read_file(const char* path) {
    std::ifstream f(path);
    if (!f.good())
        return "";

    return std::string(
        std::istreambuf_iterator<char>(f),
        std::istreambuf_iterator<char>()
    );
}

int main(int argc, char** argv) {
    if (argc != 3)
        return 1;

    std::string susp_text = read_file(argv[1]);
    std::string src_text  = read_file(argv[2]);

    auto susp_tokens = tokenize(susp_text);
    auto src_tokens  = tokenize(src_text);

    auto susp_ngrams = build_ngrams(susp_tokens, 4);
    auto src_ngrams  = build_ngrams(src_tokens, 4);

    std::sort(susp_ngrams.begin(), susp_ngrams.end(), [](const NGram& a, const NGram& b) {
        return a.hash < b.hash;
    });

    std::sort(src_ngrams.begin(), src_ngrams.end(), [](const NGram& a, const NGram& b) {
        return a.hash < b.hash;
    });

    const int MAX_OCC = 15;

    size_t i = 0;
    size_t j = 0;

    while (i < susp_ngrams.size() && j < src_ngrams.size()) {
        uint64_t h1 = susp_ngrams[i].hash;
        uint64_t h2 = src_ngrams[j].hash;

        if (h1 == h2) {
            size_t i_start = i;
            size_t j_start = j;

            while (i < susp_ngrams.size() && susp_ngrams[i].hash == h1) {
                i++;
            }

            while (j < src_ngrams.size() && src_ngrams[j].hash == h2) {
                j++;
            }

            size_t susp_count = i - i_start;
            size_t src_count  = j - j_start;

            if (susp_count <= MAX_OCC && src_count <= MAX_OCC) {
                for (size_t a = i_start; a < i; a++) {
                    for (size_t b = j_start; b < j; b++) {
                        std::cout << susp_ngrams[a].offset << " " << src_ngrams[b].offset << "\n";
                    }
                }
            }
        }
        else if (h1 < h2) {
            while (i < susp_ngrams.size() && susp_ngrams[i].hash == h1) {
                i++;
            }
        }
        else {
            while (j < src_ngrams.size() && src_ngrams[j].hash == h2) {
                j++;
            }
        }
    }

    return 0;
}