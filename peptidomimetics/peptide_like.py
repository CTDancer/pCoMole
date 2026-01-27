from cope_models.objectives import analyze_peptide_likeness

seq = '''
CNC(=O)[C@H1](CC1=CC=CC=C1)N(C)C(C)O[C@H1](CCCC=O)NCN(C)C(C)[C@H1](CC2=CC=CC=C2)N(C)C(=O)[C@H1]=O
'''
res = analyze_peptide_likeness(seq)
print(res)