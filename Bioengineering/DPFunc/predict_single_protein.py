#!/usr/bin/env python
"""
Script to run DPFunc prediction on a single PDB file.
DPFunc: Predicting protein function via deep learning with domain-guided structure information

Usage:
    python predict_single_protein.py --pdb /path/to/protein.pdb --sequence "AGKTGQM..." --name "RIXI"
"""

import os
import sys
import argparse
import pickle as pkl
import math
import numpy as np
import torch
import dgl
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

# Add DPFunc to path
sys.path.insert(0, '/mnt/home/behren23/DPFunc')

def save_pkl(file_path, val):
    with open(file_path, 'wb') as fw:
        pkl.dump(val, fw)

def read_pkl(file_path):
    with open(file_path, 'rb') as fr:
        return pkl.load(fr)

def extract_ca_coords_from_pdb(pdb_file):
    """Extract CA coordinates and sequence from PDB file.
    Handles non-standard PDB formats like Boltz output.
    """
    sequence = ""
    ca_coords = []
    
    # Map 3-letter codes to 1-letter codes
    aa_map = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
    }
    
    current_resnum = None
    current_ca = None
    current_resname = None
    
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                # Parse using split for more flexible handling
                parts = line.split()
                if len(parts) < 11:
                    continue
                
                # parts[0] = ATOM, parts[1] = atom_num, parts[2] = atom_name
                # parts[3] = resname, parts[4] = chain (may be multi-char), 
                # parts[5] = resnum, parts[6-8] = x,y,z
                atom_name = parts[2]
                resname = parts[3]
                
                # Find residue number - it should be an integer
                resnum = None
                for i in range(4, min(7, len(parts))):
                    try:
                        resnum = int(parts[i])
                        coord_start_idx = i + 1
                        break
                    except ValueError:
                        continue
                
                if resnum is None:
                    continue
                
                if resnum != current_resnum:
                    # Save previous residue if we have CA
                    if current_resnum is not None and current_ca is not None:
                        if current_resname in aa_map:
                            sequence += aa_map[current_resname]
                            ca_coords.append(current_ca)
                    current_resnum = resnum
                    current_ca = None
                    current_resname = resname
                
                if atom_name == 'CA':
                    # Parse coordinates
                    try:
                        x = float(parts[coord_start_idx])
                        y = float(parts[coord_start_idx + 1])
                        z = float(parts[coord_start_idx + 2])
                        current_ca = (x, y, z)
                    except (ValueError, IndexError):
                        pass
    
    # Don't forget last residue
    if current_resnum is not None and current_ca is not None:
        if current_resname in aa_map:
            sequence += aa_map[current_resname]
            ca_coords.append(current_ca)
    
    return sequence, ca_coords

def get_distance(point1, point2):
    """Calculate Euclidean distance between two 3D points."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))

def generate_esm_embeddings(sequence, protein_name, output_dir):
    """Generate ESM-2 embeddings for the protein sequence."""
    import esm
    
    print("Loading ESM-2 model...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    data = [(protein_name, sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    
    if torch.cuda.is_available():
        batch_tokens = batch_tokens.cuda()
    
    print("Generating embeddings...")
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    
    # Get per-residue embeddings (exclude start/end tokens)
    embeddings = results["representations"][33][0, 1:len(sequence)+1].cpu().numpy()
    
    # Save embeddings
    esm_path = os.path.join(output_dir, f"esm_{protein_name}.pkl")
    save_pkl(esm_path, {protein_name: embeddings})
    print(f"ESM embeddings saved to {esm_path}")
    
    return embeddings

def create_protein_graph(protein_name, ca_coords, esm_embeddings, threshold=12):
    """Create DGL graph from protein structure."""
    u_list = []
    v_list = []
    dis_list = []
    
    for i, coord1 in enumerate(ca_coords):
        for j, coord2 in enumerate(ca_coords):
            if i == j:
                continue
            dist = get_distance(coord1, coord2)
            if dist <= threshold:
                u_list.append(i)
                v_list.append(j)
                dis_list.append(dist)
    
    graph = dgl.graph((torch.tensor(u_list), torch.tensor(v_list)), num_nodes=len(ca_coords))
    graph.edata['dis'] = torch.tensor(dis_list, dtype=torch.float32)
    graph.ndata['x'] = torch.from_numpy(esm_embeddings).float()
    
    return graph

def get_interpro_annotations(sequence, protein_name, output_dir):
    """
    Get InterPro annotations for the protein.
    Note: This requires InterProScan or API access. For now, we create empty annotations.
    """
    # Load the InterPro index from DPFunc
    inter_idx_path = '/mnt/home/behren23/DPFunc/data/inter_idx.pkl'
    if os.path.exists(inter_idx_path):
        inter_idx = read_pkl(inter_idx_path)
        num_interpro = len(inter_idx)
    else:
        # Default size from DPFunc
        num_interpro = 26203
    
    # Create empty InterPro vector (zeros = no annotations)
    # In production, you should run InterProScan on the sequence
    interpro_vector = np.zeros(num_interpro)
    
    interpro_path = os.path.join(output_dir, f"interpro_{protein_name}.pkl")
    save_pkl(interpro_path, interpro_vector)
    print(f"InterPro annotations saved to {interpro_path}")
    print("WARNING: InterPro annotations are empty. For accurate predictions, run InterProScan on your sequence.")
    
    return interpro_vector

def prepare_prediction_files(protein_name, pdb_file, sequence, output_dir):
    """Prepare all files needed for DPFunc prediction."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Preparing DPFunc prediction for: {protein_name}")
    print(f"PDB file: {pdb_file}")
    print(f"Sequence length: {len(sequence)}")
    print(f"{'='*60}\n")
    
    # Step 1: Extract CA coordinates from PDB
    print("Step 1: Extracting CA coordinates from PDB...")
    pdb_seq, ca_coords = extract_ca_coords_from_pdb(pdb_file)
    print(f"  Extracted {len(ca_coords)} CA atoms")
    print(f"  PDB sequence: {pdb_seq[:50]}...")
    
    # Save coordinates
    save_pkl(os.path.join(output_dir, 'pdb_points.pkl'), {protein_name: ca_coords})
    save_pkl(os.path.join(output_dir, 'pdb_seqs.pkl'), {protein_name: pdb_seq})
    
    # Step 2: Generate ESM embeddings
    print("\nStep 2: Generating ESM-2 embeddings...")
    esm_embeddings = generate_esm_embeddings(pdb_seq, protein_name, output_dir)
    
    # Step 3: Create protein graph
    print("\nStep 3: Creating protein structure graph...")
    graph = create_protein_graph(protein_name, ca_coords, esm_embeddings)
    graph_path = os.path.join(output_dir, f'graph_{protein_name}.pkl')
    save_pkl(graph_path, [graph])
    print(f"  Graph has {graph.num_nodes()} nodes and {graph.num_edges()} edges")
    
    # Step 4: Get InterPro annotations
    print("\nStep 4: Preparing InterPro annotations...")
    interpro = get_interpro_annotations(pdb_seq, protein_name, output_dir)
    
    # Step 5: Create protein list file
    pid_list_path = os.path.join(output_dir, 'pid_list.pkl')
    save_pkl(pid_list_path, [protein_name])
    
    # Step 6: Create dummy GO file (empty for prediction)
    go_path = os.path.join(output_dir, 'protein_go.txt')
    with open(go_path, 'w') as f:
        f.write(f"{protein_name}\t\n")  # Empty GO terms for prediction
    
    print(f"\n{'='*60}")
    print("Preparation complete!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")
    
    return {
        'pid_list_file': pid_list_path,
        'pid_go_file': go_path,
        'pid_pdb_file': graph_path,
        'interpro_file': os.path.join(output_dir, f"interpro_{protein_name}.pkl"),
    }

def main():
    parser = argparse.ArgumentParser(description='Prepare single protein for DPFunc prediction')
    parser.add_argument('--pdb', required=True, help='Path to PDB file')
    parser.add_argument('--sequence', help='Protein sequence (optional, extracted from PDB if not provided)')
    parser.add_argument('--name', default='protein', help='Protein name/ID')
    parser.add_argument('--output', default='./dpfunc_prediction', help='Output directory')
    
    args = parser.parse_args()
    
    # Use provided sequence or extract from PDB
    if args.sequence:
        sequence = args.sequence
    else:
        sequence, _ = extract_ca_coords_from_pdb(args.pdb)
    
    # Prepare files
    files = prepare_prediction_files(args.name, args.pdb, sequence, args.output)
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("""
To run DPFunc prediction, you need to:

1. Download the pre-trained model from:
   https://drive.google.com/file/d/1V0VTFTiB29ilbAIOZn0okBQWPlbOI3wN/view

2. Modify the configuration file (e.g., configure/mf.yaml) to point to your files:

   test:
     name: test
     pid_list_file: {pid_list}
     pid_go_file: {go_file}
     pid_pdb_file: {pdb_file}
     interpro_file: {interpro}

3. Run prediction:
   python DPFunc_pred.py -d mf -n 0 -p my_prediction

Note: For accurate InterPro annotations, run InterProScan on your sequence:
   https://www.ebi.ac.uk/interpro/search/sequence/
""".format(
        pid_list=files['pid_list_file'],
        go_file=files['pid_go_file'],
        pdb_file=files['pid_pdb_file'],
        interpro=files['interpro_file']
    ))

if __name__ == '__main__':
    main()
