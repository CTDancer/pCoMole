import os
import shutil

def combine_to_complex(res_path, vina_path, output_path):
    """Combines two PDBs into one, assigning Chain A to docking and Chain B to receptor."""
    with open(output_path, 'w') as complex_file:
        # Process best_docked1.pdb as Chain A
        with open(res_path, 'r') as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    # Column 22 (index 21) is the Chain ID in PDB format
                    line = line[:21] + "A" + line[22:]
                    complex_file.write(line)
        complex_file.write("TER\n") 

        # Process receptor.pdb as Chain B
        with open(vina_path, 'r') as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    line = line[:21] + "B" + line[22:]
                    complex_file.write(line)
        complex_file.write("TER\n")
        complex_file.write("END\n")

def consolidate_files(base_path, output_folder):
    results_dir = os.path.join(base_path, 'results')
    vina_prep_dir = os.path.join(base_path, 'vina_prep')
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Filter for directories that exist in 'results'
    target_folders = [f for f in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, f))]

    for folder_name in target_folders:
        res_source = os.path.join(results_dir, folder_name, 'best_docked1.pdb')
        vina_source = os.path.join(vina_prep_dir, folder_name, 'receptor_prep.pdb')
        dest_subfolder = os.path.join(output_folder, folder_name)
        
        # Only proceed if both source files exist
        if os.path.exists(res_source) and os.path.exists(vina_source):
            os.makedirs(dest_subfolder, exist_ok=True)
            
            # 1. Copy original best_docked1.pdb
            shutil.copy2(res_source, os.path.join(dest_subfolder, 'best_docked1.pdb'))
            
            # 2. Copy original receptor.pdb
            shutil.copy2(vina_source, os.path.join(dest_subfolder, 'receptor_prep.pdb'))
            
            # 3. Create the combined complex file
            complex_filename = f"{folder_name}_complex.pdb"
            combine_to_complex(res_source, vina_source, os.path.join(dest_subfolder, complex_filename))
            
            print(f"Done: {folder_name} (3 files created)")
        else:
            print(f"Skipping {folder_name}: Missing one or more .pdb files.")

# Execute
consolidate_files(base_path='./batch2', output_folder='./batch2/complexes')