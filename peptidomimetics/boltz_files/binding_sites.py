from Bio.PDB import MMCIFParser, NeighborSearch

def get_binding_residues(cif_file, ligand_chain_id='A', protein_chain_id='B', distance_cutoff=6.0):
    # 1. Parse the structure
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("complex", cif_file)
    model = structure[0]

    # 2. Separate atoms into ligand and protein groups
    try:
        ligand_atoms = list(model[ligand_chain_id].get_atoms())
        protein_atoms = list(model[protein_chain_id].get_atoms())
    except KeyError as e:
        return f"Error: Chain {e} not found in the CIF file."

    if not ligand_atoms or not protein_atoms:
        return "Error: One of the chains is empty."

    # 3. Setup NeighborSearch with protein atoms
    # This creates a spatial index (KDTree) for fast lookup
    searcher = NeighborSearch(protein_atoms)

    # 4. Find protein residues near ligand atoms
    binding_residues = set()
    
    for atom in ligand_atoms:
        # Find all protein atoms within the distance cutoff of this specific ligand atom
        nearby_atoms = searcher.search(atom.coord, distance_cutoff)
        
        for protein_atom in nearby_atoms:
            # Get the parent residue object
            residue = protein_atom.get_parent()
            binding_residues.add(residue)

    # 5. Sort and return results
    # Sorting by residue ID for readability
    sorted_residues = sorted(list(binding_residues), key=lambda x: x.id[1])
    
    return sorted_residues

# --- Execution ---
file_path = "/scratch/pranamlab/tong/pCoMol/peptidomimetics/boltz_files/boltz_results_7JVS/predictions/16/16_model_0.cif" # Replace with your filename
results = get_binding_residues(file_path)

if isinstance(results, list):
    print(f"Found {len(results)} binding residues in Chain B within 6Å of Chain A:\n")
    print(f"{'Residue':<10} | {'ID':<10}")
    print("-" * 22)
    for res in results:
        res_name = res.get_resname()
        res_id = res.id[1]
        print(f"{res_name:<10} | {res_id:<10}")
else:
    print(results)