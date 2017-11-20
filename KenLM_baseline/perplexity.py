import kenlm
import numpy as np
import math

m = kenlm.Model("3-gram.arpa")

per_file = "../penn/test.txt"
data = []
for line in open(per_file, 'r'):
    for word in line.split():
        #if word != "<unk>":
        data.append(word)
s = " ".join(data)

perpl = m.perplexity(s) #This should work with kenlm.Model. But this is not working now...
print(perpl)

#cat train.txt| python ../KenLM/process.py | ../../../../../kenlm/bin/lmplz -o 3 > train.arpa
