//computes the encoplot data of a pair of files
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <regex>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

typedef __int128 tngram;

//CrG rsort
#define fr(x,y) for(int x = 0; x < y; x++)

int* index_rsort_ngrams(unsigned char *x,int l,int DEPTH) {
    int NN = l - DEPTH + 1; 
    if(NN > 0) {
        unsigned char *pin = x + NN;
        unsigned char *pout = x;
        int *ix = (int*) malloc(NN * sizeof(int));
        int *ox = (int*) malloc(NN * sizeof(int));
        const int RANGE = 256;
        int counters[RANGE]; 
        int startpos[RANGE];
        fr(i,NN) ix[i] = i;

        // radix sort,the input is x,
        // the output rank is ix
        fr(k,RANGE) counters[k] = 0;
        fr(i,NN) counters[*(x+i)]++;
        fr(j,DEPTH) {
            int ofs = j; //low endian
            int sp = 0;       
            fr(k,RANGE) {
                startpos[k] = sp;
                sp += counters[k];
            }
            fr(i,NN) {
                unsigned char c = x[ofs+ix[i]];
                ox[startpos[c]++] = ix[i];
            }
            memcpy(ix, ox, NN * sizeof(ix[0]));
            
            //update counters
            if(j < DEPTH - 1) {
                counters[*pout++]--;
                counters[*pin++]++;
            }
        }
        free(ox);

        return ix;
    }
}

#define MAXBUFSIZ 8000123
unsigned char file1[MAXBUFSIZ];
unsigned char file2[MAXBUFSIZ];
int l1,l2;

inline tngram readat(const unsigned char *buf, int poz){
    return *(tngram*)(buf + poz);
}

int main(int argc,char** argv) {
    int depth = sizeof(tngram);
    FILE* f1 = fopen(argv[1],"rb");

    l1 = fread(file1, 1, MAXBUFSIZ, f1);
    fclose(f1);
    FILE* f2=fopen(argv[2], "rb");
    l2 = fread(file2, 1, MAXBUFSIZ, f2);
    fclose(f2);

    //index the ngrams
    int *ix1 = index_rsort_ngrams(file1, l1, depth);
    int *ix2 = index_rsort_ngrams(file2, l2, depth);
    int i1 = 0;
    int i2 = 0; //merge
    tngram s1 = readat(file1, ix1[i1]);
    tngram s2 = readat(file2, ix2[i2]);
    l1 -= (depth - 1);
    l2 -= (depth - 1);
    while(i1 < l1 && i2 < l2) {
        if(s1 == s2) {
            printf("%d-%d,", ix1[i1], ix2[i2]);
        i1++;
        if(i1 < l1) s1 = readat(file1,ix1[i1]);

        i2++;
        if(i2 < l2)
            s2 = readat(file2,ix2[i2]);
        }
        else if(s1 < s2) {
            i1++;
            if(i1 < l1)
                s1 = readat(file1, ix1[i1]);
        }
        else if(s2 < s1) {
            i2++;
            if(i2 < l2)
                s2 = readat(file2,ix2[i2]);
        }
    }   
    free(ix2);
    free(ix1);

    return 0;
}