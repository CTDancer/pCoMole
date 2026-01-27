from constraints import GFP
from pathlib import Path

def read_fasta_sequences(fasta_path: str):
    """
    Parse a FASTA file and return:
      - seq_ids: list of header IDs (everything after '>' up to first whitespace)
      - seqs:    list of sequences (uppercase, no spaces)
    """
    fasta_path = Path(fasta_path)
    seq_ids, seqs = [], []

    cur_id = None
    cur_chunks = []

    with fasta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                # flush previous record
                if cur_id is not None:
                    seq_ids.append(cur_id)
                    seqs.append("".join(cur_chunks).replace(" ", "").upper())

                # start new record
                header = line[1:].strip()
                cur_id = header.split()[0]  # ID token
                cur_chunks = []
            else:
                cur_chunks.append(line)

    # flush last record
    if cur_id is not None:
        seq_ids.append(cur_id)
        seqs.append("".join(cur_chunks).replace(" ", "").upper())

    return seq_ids, seqs


gfp = GFP('cuda:0')
_, seqs = read_fasta_sequences('./gfps.fasta')

res = gfp(seqs)
print(len(seqs))
print(sum(res))