from constraints import GFP

with open('/scratch/pranamlab/tong/pCoMol/gfp/FPredX/raygun.fasta', 'r') as f:
    lines = f.readlines()

seqs = [seq.strip() for seq in lines if '>' not in seq]

gfp_classifier = GFP('cuda:0')
gfp_probs = gfp_classifier.get_scores(seqs, return_probs=False)
print(sum(gfp_probs))