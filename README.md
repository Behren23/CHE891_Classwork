# CHE891_Classwork

Classwork related to CHE891/CMSE882 — Machine Learning in Bioengineering.

This repository contains tools and workflows for **protein structure prediction**, **protein function prediction**, **binding affinity estimation**, and **conformational ensemble generation**. All heavy computation is designed to run on an HPC cluster with GPU access (e.g., SLURM with V100 GPUs).

---

## Table of Contents

1. [Boltz-2 — Structure & Affinity Prediction](#1-boltz-2--structure--affinity-prediction)
2. [DPFunc — Protein Function Prediction](#2-dpfunc--protein-function-prediction)
3. [OnionNet — Protein-Ligand Binding Affinity (CNN)](#3-onionnet--protein-ligand-binding-affinity-cnn)
4. [aSAM (sam2) — Conformational Ensemble Generation](#4-asam-sam2--conformational-ensemble-generation)
5. [BioEmu — Protein Equilibrium Ensemble Sampling](#5-bioemu--protein-equilibrium-ensemble-sampling)
6. [Analysis Scripts](#6-analysis-scripts)
7. [Repository Structure](#7-repository-structure)

---

## 1. Boltz-2 — Structure & Affinity Prediction

**Location:** `Bioengineering/boltz2/`

Boltz-2 is a biomolecular foundation model that predicts protein–ligand complex structures and binding affinities. It approaches the accuracy of physics-based FEP methods while running 1000× faster.

- **Paper:** [Boltz-2 Technical Report](https://doi.org/10.1101/2025.06.14.659707)
- **GitHub:** <https://github.com/jwohlwend/boltz>

### 1.1 Installation

```bash
pip install boltz[cuda] -U
```

### 1.2 Structure Prediction

```bash
# Request a GPU session
salloc --gpus=v100:1 --mem=64G --time=03:00:00

# Activate the environment
source boltz/.venv_clean/bin/activate

# Predict a single protein structure
boltz predict boltz/1_RIXI.yaml --use_msa_server --output_format pdb
```

### 1.3 Binding Affinity Prediction

Prepare a YAML file containing a protein sequence and a ligand SMILES string:

```yaml
version: 1
sequences:
  - protein:
      id: P
      sequence: YOUR_PROTEIN_SEQUENCE_HERE
      msa: empty
  - ligand:
      id: L
      smiles: 'YOUR_SMILES_STRING_HERE'
properties:
```

Run the prediction:

```bash
boltz predict my_ligand.yaml --use_msa_server --output_format pdb
```

Batch prediction over all YAML files in a folder:

```bash
for yaml in "ligand yaml files"/*.yaml; do
    boltz predict "$yaml" --use_msa_server --output_format pdb
done
```

### 1.4 Output

Results are saved in `boltz_results_<name>/predictions/<name>/`:

| File | Description |
|------|-------------|
| `affinity_<name>.json` | Predicted binding affinity values |
| `<name>_model_0.pdb` | 3D structure of the complex |
| `confidence_<name>_model_0.json` | Model confidence scores |

**Affinity JSON fields:**

| Field | Use |
|-------|-----|
| `affinity_pred_value` | Predicted pKd (`log10(IC50)` in µM). Use for lead optimization. |
| `affinity_probability_binary` | Probability of binding (0–1). Use for hit discovery. |

### 1.5 Extract All Affinity Results

```bash
python3 << 'EOF'
import json, pandas as pd
from pathlib import Path

results = []
for f in Path('.').rglob('affinity_*.json'):
    with open(f) as fp:
        data = json.load(fp)
    name = f.stem.replace('affinity_', '')
    results.append({
        'Ligand': name,
        'pKd': data.get('affinity_pred_value', 0),
        'Binding_Prob': data.get('affinity_probability_binary', 0),
    })
df = pd.DataFrame(results).sort_values('pKd', ascending=False)
df.to_excel('binding_affinity_results.xlsx', index=False)
print(df.to_string(index=False))
EOF
```

---

## 2. DPFunc — Protein Function Prediction

**Location:** `Bioengineering/DPFunc/`

DPFunc predicts Gene Ontology (GO) protein functions using deep learning with domain-guided structure information.

- **Paper:** Wang et al., *Nature Communications* 2025
- **GitHub:** <https://github.com/CSUBioGroup/DPFunc>

### 2.1 Environment Setup

```bash
# Request a GPU session
salloc --gpus=v100:1 --mem=64G --time=03:00:00

module purge
export PYTHONPATH=""
cd ~/DPFunc
source dpfunc_env/bin/activate

# Verify
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); import dgl; print('DGL:', dgl.__version__)"
```

### 2.2 Data Preparation

Follow the Jupyter notebook tutorial:

```bash
jupyter notebook DataProcess/Process_data.ipynb
```

Or download pre-processed data from the links in `data/download_link.txt`.

### 2.3 Training

```bash
# Train on Molecular Function (mf), Biological Process (bp), or Cellular Component (cc)
python DPFunc_main.py -d mf -n 0 -e 15 -p my_model
```

**Arguments:**

| Flag | Description | Default |
|------|-------------|---------|
| `-d` | Ontology type (`mf` / `bp` / `cc`) | — |
| `-n` | GPU number | `0` |
| `-e` | Training epochs | `15` |
| `-p` | Output prefix | `temp_model` |

### 2.4 Prediction (Pre-trained Model)

```bash
python DPFunc_pred.py -d mf -n 0 -p my_prediction
```

### 2.5 Single Protein Prediction (from a PDB file)

This is useful for predicting the function of a Boltz-generated structure:

```bash
# Step 1: Prepare protein data (generates ESM embeddings, structure graph, etc.)
python predict_single_protein.py \
    --pdb /path/to/protein.pdb \
    --name "RIXI" \
    --output ./dpfunc_prediction_RIXI

# Step 2: Update configure/mf.yaml test section to point to generated files, then:
python DPFunc_pred.py -d mf -n 0 -p RIXI_prediction
```

---

## 3. OnionNet — Protein-Ligand Binding Affinity (CNN)

**Location:** `Bioengineering/onionnet/`

OnionNet uses multi-layer inter-molecular contact features with a CNN to predict protein–ligand binding affinity (pKa).

- **Paper:** Zheng et al., *arXiv:1906.02418*
- **GitHub:** <https://github.com/zhenglz/onionnet>

### 3.1 Environment Setup

```bash
source /etc/profile.d/modules.sh
module load Python/3.11.3
cd ~/onionnet
source .venv/bin/activate
```

Or create from the provided environment file:

```bash
conda env create -f onet_env.yaml
conda activate onionnet
```

### 3.2 Prepare Protein-Ligand Complexes

Protein and ligand must be in the **same PDB file** with the ligand residue name set to `LIG`:

```bash
# Convert ligand to PDB if needed
obabel ligand.mol2 -O ligand.pdb

# Combine protein + ligand
bash tools/prepare_complex.sh protein.pdb ligand.pdb complex.pdb
```

### 3.3 Create Input File

Create a text file listing complex PDB paths (one per line):

```
complex1/complex1.pdb
complex2/complex2.pdb
```

### 3.4 Generate Features & Predict

```bash
# Generate contact features
python generate_features.py -inp input_complexes.dat -out features.csv

# Predict binding affinity
python predict.py -fn features.csv -out predicted_pKa.csv \
    -weights models/CNN_final_model_weights.h5 \
    -scaler models/StandardScaler_new.model
```

### 3.5 Using with Boltz Output

```bash
# Boltz PDBs already contain protein + ligand — list them as input
ls boltz_results_*/predictions/*/*.pdb > boltz_complexes.dat

python generate_features.py -inp boltz_complexes.dat -out boltz_features.csv
python predict.py -fn boltz_features.csv -out boltz_pKa.csv \
    -weights models/CNN_final_model_weights.h5 \
    -scaler models/StandardScaler_new.model
```

### 3.6 Interpreting Output

| pKa Range | Binding Strength |
|-----------|-----------------|
| ~10 | Very strong (nM) |
| ~7 | Strong (µM) |
| ~5 | Moderate |
| ~3 | Weak |

### 3.7 Tutorial Test

```bash
cd tutorials/PDB_samples
python ../../generate_features.py -inp input_PDB_testing.dat -out features.csv
python ../../predict.py -fn features.csv -out predictions.csv \
    -weights ../../models/CNN_final_model_weights.h5 \
    -scaler ../../models/StandardScaler_new.model
cat predictions.csv
```

---

## 4. aSAM (sam2) — Conformational Ensemble Generation

**Location:** `Bioengineering/sam2/`

aSAM is a latent diffusion model for generating heavy-atom protein conformational ensembles, trained on MD simulation data (mdCATH / ATLAS).

- **Paper:** [aSAM Preprint](https://www.biorxiv.org/content/10.1101/2025.03.09.642148v1)
- **GitHub:** <https://github.com/giacomo-janson/sam2>

### 4.1 Installation

```bash
conda create --name sam2 python=3.10
conda activate sam2
# Install PyTorch with CUDA first (see https://pytorch.org)
pip install -e .
```

### 4.2 Generate Ensembles

**mdCATH-based aSAMt** (temperature-conditioned):

```bash
python scripts/generate_ensemble.py \
    -c config/mdcath_model.yaml \
    -i examples/input/4qbuA03.320.pdb \
    -o protein -n 250 -b 8 -T 320 -d cuda
```

**ATLAS-based aSAMc** (no temperature conditioning):

```bash
python scripts/generate_ensemble.py \
    -c config/atlas_model.yaml \
    -i examples/input/6h49_A.pdb \
    -o protein -n 250 -b 8 -d cuda
```

**Key arguments:**

| Flag | Description | Default |
|------|-------------|---------|
| `-c` | Config YAML file | — |
| `-i` | Input PDB file | — |
| `-o` | Output path prefix | — |
| `-n` | Number of conformations | `250` |
| `-b` | Batch size | `8` |
| `-T` | Temperature in Kelvin (mdCATH only) | — |
| `-d` | Device (`cuda` / `cpu`) | `cuda` |

**Input requirements:** single-chain PDB, no missing heavy atoms, standard amino acids only.

---

## 5. BioEmu — Protein Equilibrium Ensemble Sampling

**Location:** `Bioengineering/bioemu/`

BioEmu (Biomolecular Emulator) is a generative model from Microsoft Research that samples from the approximated equilibrium distribution of protein structures given an amino acid sequence. It generates backbone conformational ensembles that approximate molecular dynamics simulations.

- **Paper:** Lewis et al., *Science* 2025 — [DOI:10.1126/science.adv9817](https://www.science.org/doi/10.1126/science.adv9817)
- **Model Weights:** [HuggingFace (microsoft/bioemu)](https://huggingface.co/microsoft/bioemu)
- **GitHub:** <https://github.com/microsoft/bioemu>

### 5.1 Installation

```bash
pip install bioemu
```

> **Note:** The first time BioEmu runs, it will set up [ColabFold](https://github.com/sokrypton/ColabFold) in `~/.bioemu_colabfold` for MSA/embedding generation. Set `BIOEMU_COLABFOLD_DIR` to change the location.

### 5.2 Basic Sampling

```bash
# Request a GPU session
salloc --gpus=v100:1 --mem=64G --time=03:00:00

# Quick test with a small peptide
python -m bioemu.sample --sequence GYDPETGTWG --num_samples 10 --output_dir ~/test-chignolin
```

Or pass a FASTA file:

```bash
python -m bioemu.sample \
    --sequence sequence.fasta \
    --num_samples 50 \
    --output_dir ~/bioemu_results/my_protein
```

Python API:

```python
from bioemu.sample import main as sample
sample(sequence='GYDPETGTWG', num_samples=10, output_dir='~/test_chignolin')
```

Model weights are automatically downloaded from HuggingFace on first use.

### 5.3 Steering (Reduce Unphysical Structures)

BioEmu includes a Sequential Monte Carlo (SMC) steering system to reduce chain breaks and steric clashes:

```bash
python -m bioemu.sample \
    --sequence GYDPETGTWG \
    --num_samples 100 \
    --output_dir ~/steered-samples \
    --steering_config src/bioemu/config/steering/physical_steering.yaml \
    --denoiser_config src/bioemu/config/denoiser/stochastic_dpm.yaml
```

**Available steering potentials:**

| Potential | Description |
|-----------|-------------|
| ChainBreak | Prevents backbone discontinuities |
| ChainClash | Avoids steric clashes between non-neighboring residues |
| DisulfideBridge | Encourages disulfide bond formation between cysteine pairs |

### 5.4 Side-Chain Reconstruction & MD Relaxation

```bash
# Install optional dependencies
pip install bioemu[md]

# Reconstruct side-chains and run energy minimization
python -m bioemu.sidechain_relax --pdb-path path/to/topology.pdb --xtc-path path/to/samples.xtc
```

Options:
- `--no-md-equil` — side-chain reconstruction only, skip MD
- `--md-protocol nvt_equil` — run a short 0.1 ns NVT equilibration

### 5.5 Model Checkpoints

| Checkpoint | Description |
|------------|-------------|
| `bioemu-v1.0` | Preprint weights |
| `bioemu-v1.1` | Published *Science* paper weights (default) |
| `bioemu-v1.2` | Extended training with experimental folding free energies |

Select a specific checkpoint:

```bash
python -m bioemu.sample --model_name bioemu-v1.1 --sequence ... --num_samples 50 --output_dir ...
```

### 5.6 Output

Results are saved to the specified `--output_dir`:

| File | Description |
|------|-------------|
| `samples.xtc` | Trajectory of sampled conformations |
| `sequence.fasta` | Input protein sequence |
| `topology.pdb` | Topology for parsing the XTC file |
| `batch_*.npz` | Raw sample batches (backbone frames) |

By default, unphysical structures (clashes/chain breaks) are filtered out. Use `--filter_samples=False` to keep all.

### 5.7 Sampling Times (A100 GPU, 80 GB VRAM)

| Sequence Length | Time (1000 samples) |
|----------------:|--------------------:|
| 100 | ~4 min |
| 300 | ~40 min |
| 600 | ~150 min |

> **Note:** Very long sequences (>600 residues) may require `batch_size=1` and significantly more time.

---

## 6. Analysis Scripts

**Location:** `Bioengineering/scripts/`

### 5.1 Ensemble Analysis (vs. Reference)

Computes folded state fraction, secondary structure preservation, and initRMSD:

```bash
python scripts/ensemble_analysis.py \
    -n native.pdb \
    -p ensemble_topology.pdb \
    -t ensemble_trajectory.dcd \
    --q_thresh 0.6
```

### 5.2 Ensemble Comparison (Two Ensembles)

Computes PCC Cα RMSF, χ-JSD, heavy clashes, and peptide bond violations:

```bash
python scripts/ensemble_comparison.py \
    -P ref_topology.pdb -T ref_trajectory.dcd \
    -p hat_topology.pdb -t hat_trajectory.dcd \
    -i initial.pdb
```

---

## 7. Repository Structure

```
CHE891_Classwork/
├── Bioengineering/
│   ├── boltz2/              # Boltz-2 structure & affinity prediction
│   │   ├── *.yaml           # Protein input files
│   │   ├── ligand yaml files/  # PEG ligand YAML files for affinity
│   │   ├── OPERATION_STRUCTURE.txt
│   │   └── OPERATION_AFFINITY.txt
│   ├── DPFunc/              # DPFunc protein function prediction
│   │   ├── DPFunc_main.py   # Training script
│   │   ├── DPFunc_pred.py   # Prediction script
│   │   ├── predict_single_protein.py  # Single PDB prediction
│   │   ├── configure/       # YAML configs (mf/bp/cc)
│   │   └── OPERATION.txt
│   ├── onionnet/            # OnionNet binding affinity (CNN)
│   │   ├── generate_features.py
│   │   ├── predict.py
│   │   ├── models/          # Pre-trained CNN weights
│   │   └── OPERATION_ONIONNET.txt
│   ├── sam2/                # aSAM conformational ensemble generation
│   │   ├── config/          # Model YAML configs
│   │   ├── scripts/         # Inference scripts
│   │   └── sam/             # Core library
│   ├── bioemu/              # BioEmu equilibrium ensemble sampling
│   │   └── src/bioemu/      # Core library & configs
│   ├── bioemur__results/    # BioEmu ensemble results (XTC/NPZ)
│   └── scripts/             # Analysis & comparison scripts
└── README.md
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `libpython3.11.so.1.0: cannot open shared object` | `module load Python/3.11.3` |
| MSA server timeout (Boltz) | Retry, or remove `--use_msa_server` for local MSA |
| `No module named 'sklearn.preprocessing.data'` (OnionNet) | Use `StandardScaler_new.model` |
| GPU not detected | Verify with `nvidia-smi`; request GPU via `salloc --gpus=v100:1` |
| `No module named 'openpyxl'` | `pip install openpyxl` |
