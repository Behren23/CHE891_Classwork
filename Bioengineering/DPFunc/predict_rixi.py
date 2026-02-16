#!/usr/bin/env python
"""
Simplified DPFunc prediction for a single protein.
"""

import os
import sys
import warnings
import pickle as pkl
import numpy as np
import torch
import joblib
from pathlib import Path
from scipy.sparse import csr_matrix

# Add DPFunc to path
sys.path.insert(0, '/mnt/home/behren23/DPFunc')

from DPFunc.models import combine_inter_model
from dgl.dataloading import GraphDataLoader

def predict_single_protein(protein_name, graph_file, interpro_file, ont='mf', gpu_number=0, use_cpu=False):
    """
    Run DPFunc prediction on a single protein.
    
    Args:
        protein_name: Name of the protein
        graph_file: Path to the protein graph pickle file
        interpro_file: Path to the interpro pickle file
        ont: Ontology type ('mf', 'bp', 'cc')
        gpu_number: GPU device number
        use_cpu: Force CPU usage
    """
    if use_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{gpu_number}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the multi-label binarizer
    mlb_path = f'./mlb/{ont}_go.mlb'
    print(f"Loading MLB from {mlb_path}")
    mlb = joblib.load(Path(mlb_path))
    labels_num = len(mlb.classes_)
    print(f"Number of GO term classes: {labels_num}")
    
    # Create idx_goid mapping
    idx_goid = {idx: goid for idx, goid in enumerate(mlb.classes_)}
    
    # Load the protein graph
    print(f"Loading graph from {graph_file}")
    with open(graph_file, 'rb') as fr:
        graphs = pkl.load(fr)
    graph = graphs[0]
    print(f"Graph loaded: {graph.num_nodes()} nodes, {graph.num_edges()} edges")
    
    # Load interpro features
    print(f"Loading interpro from {interpro_file}")
    with open(interpro_file, 'rb') as fr:
        interpro_vec = pkl.load(fr)
    
    # Convert to sparse matrix format expected by DPFunc
    if isinstance(interpro_vec, np.ndarray):
        interpro_matrix = csr_matrix(interpro_vec.reshape(1, -1))
    else:
        interpro_matrix = interpro_vec
    
    inter_size = interpro_matrix.shape[1]
    print(f"InterPro feature size: {inter_size}")
    
    # Create dummy labels (zeros for prediction)
    dummy_y = np.zeros((1, labels_num), dtype=np.float32)
    
    # Create dataloader
    test_data = [(graph, 0, dummy_y[0])]
    test_dataloader = GraphDataLoader(
        test_data,
        batch_size=1,
        drop_last=False,
        shuffle=False
    )
    
    # Initialize model
    print("Initializing model...")
    model = combine_inter_model(
        inter_size=inter_size,
        inter_hid=1280,
        graph_size=1280,
        graph_hid=1280,
        label_num=labels_num,
        head=4
    ).to(device)
    
    # Load and run all 3 model checkpoints
    all_predictions = []
    
    for i in range(3):
        model_path = f'./save_models/DPFunc_model_{ont}_{i}of3model.pt'
        if os.path.exists(model_path):
            print(f"Loading model {i+1}/3: {model_path}")
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Run prediction
            with torch.no_grad():
                for batch_graph, batch_idx, _ in test_dataloader:
                    batch_graph = batch_graph.to(device)
                    
                    # Get node features (ESM embeddings)
                    feats = batch_graph.ndata['x']
                    
                    # Get interpro features for this batch as sparse format (indices, indptr, data)
                    batch_inter_sparse = interpro_matrix[batch_idx.numpy()]
                    inter_features = (
                        torch.from_numpy(batch_inter_sparse.indices).to(device).long(),
                        torch.from_numpy(batch_inter_sparse.indptr).to(device).long(),
                        torch.from_numpy(batch_inter_sparse.data).to(device).float()
                    )
                    
                    # Forward pass: model(inter_features, graph, graph_node_features)
                    output = model(inter_features, batch_graph, feats)
                    pred = torch.sigmoid(output).cpu().numpy()
                    all_predictions.append(pred)
        else:
            print(f"Warning: Model not found: {model_path}")
    
    if not all_predictions:
        print("Error: No models loaded!")
        return None
    
    # Average predictions from all models
    avg_pred = np.mean(all_predictions, axis=0)[0]  # Shape: (labels_num,)
    
    # Get top predictions
    print("\n" + "="*60)
    print(f"TOP PREDICTED GO TERMS for {protein_name}")
    print("="*60)
    
    # Sort by score
    sorted_indices = np.argsort(avg_pred)[::-1]
    
    results = []
    for rank, idx in enumerate(sorted_indices[:20], 1):
        go_term = idx_goid[idx]
        score = avg_pred[idx]
        results.append((go_term, score))
        print(f"{rank:2d}. {go_term}: {score:.4f}")
    
    # Save results
    results_file = f'./results/{protein_name}_{ont}_predictions.pkl'
    os.makedirs('./results', exist_ok=True)
    with open(results_file, 'wb') as fw:
        pkl.dump({
            'protein': protein_name,
            'ontology': ont,
            'predictions': dict(zip([idx_goid[i] for i in range(labels_num)], avg_pred)),
            'top_20': results
        }, fw)
    print(f"\nResults saved to: {results_file}")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='DPFunc single protein prediction')
    parser.add_argument('--name', default='RIXI', help='Protein name')
    parser.add_argument('--graph', default='./dpfunc_prediction_RIXI/graph_RIXI.pkl', help='Path to graph file')
    parser.add_argument('--interpro', default='./dpfunc_prediction_RIXI/interpro_RIXI.pkl', help='Path to interpro file')
    parser.add_argument('--ont', default='mf', choices=['mf', 'bp', 'cc'], help='Ontology type')
    parser.add_argument('--gpu', type=int, default=0, help='GPU number')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    
    args = parser.parse_args()
    
    results = predict_single_protein(
        protein_name=args.name,
        graph_file=args.graph,
        interpro_file=args.interpro,
        ont=args.ont,
        gpu_number=args.gpu,
        use_cpu=args.cpu
    )
