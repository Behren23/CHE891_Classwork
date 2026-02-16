$ python Untitled-1.py
$ ml spider AlphaFold
$ ml spider AlphaFold/<version>
$ getexample AlphaFold
ml spider AlphaFold
getexample AlphaFold
help
module load powertools
boltz predict input_path --use_msa_server
predict input_path --use_msa_server
help
pip install boltz[cuda] -U
/mnt/ffs24/home/behren23/boltz/.venv/bin/python -m pip install boltz[cuda] -U
$ getexample AlphaFold
predict input_path --use_msa_server
boltz predict input_path --use_msa_server
oltz predict --help
boltz predict --help
cd boltz; pip install -e .[cuda]
ml spider AlphaFold
$ ml spider AlphaFold
$ ml spider AlphaFold
ml spider AlphaFold
help
cd AlphaFold
source /mnt/ffs24/home/behren23/boltz/.venv/bin/activate
.boltz_env\Scripts\activate
boltz predict
source /mnt/ffs24/home/behren23/boltz/.venv/bin/activate
boltz predict 1_RIKI.yaml 
boltz predict --help
boltz predict --out_dir PATH
boltz predict --cache PATH
PATH
boltz predict 1_RIXI.yaml
boltz predict 1_RIXI.yaml --use_msa_server
boltz predict 1_RIKI.yaml --use_msa_server --out_dir ./boltz_results
boltz predict 1_RIKI.yaml --use_msa_server
boltz predict 1_RIXI.yaml --use_msu_server
boltz predict 1_RIXI.yaml --use_msa_server
boltz predict 2_PGIP2_8IKW.yaml --use_msa_server
boltz predict 1_RIXI.yaml --accelerator cpu
boltz predict 1_RIXI.yaml --accelerator cpu --use_msa_server
cd AlphaFold
salloc --gpus=1 --mem=32G --time=02:00:00
python -m venv .boltz_env
help
source .venv_clean/bin/activate
boltz predict boltz/4_FLS2_4MNA.yaml
boltz/4_FLS2_4MNA.yamlboltz predict 4_FLS2_4MNA.yaml
boltz predict 4_FLS2_4MNA.yaml
boltz predict 4_FLS2_4MNA.yaml --use_msa_server
nvidia-smi
pip install cuequivariance-torch
pip install cuequivariance-ops-torch-cu12
boltz predict 4_FLS2_4MNA.yaml --use_msa_server
boltz predict 1_RIXI.yaml --use_msa_server
boltz predict 2_PGIP2_8IKW.yaml --use_msa_server
boltz predict 3_OATP1B1_8HND.yaml --use_msa_server
module purge
module load Python/3.11.3  # Or your cluster's latest Python
python -m venv ~/colabfold_env
source ~/colabfold_env/bin/activate
# Install ColabFold
pip install "colabfold[alphafold-minus-jax] @ git+https://github.com/sokrypton/ColabFold"
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
colabfold_batch 1_RIXI.fasta ./af2_results
colabfold_batch colabfold_env/1_RIXI.fasta ./af2_results
nvidia-smi
source ~/colabfold_env/bin/activate
ls -F
colabfold_batch inputs/1_RIXI.fasta ./af2_results
cd ~/collabfold_env
cd ~/colabfold_env
colabfold_batch 1_RIXI.fasta ./af2_results
nvidia-smi
pip uninstall jax jaxlib -y
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install --upgrade pip
python -c "import jax; from jax.lib import xla_extension; print('JAX fixed')"
exit
source ~/colabfold_env/bin/activate
cd ~/collabfold_env
cd ~/colabfold_env
source ~/colabfold_env/bin/activate
colabfold_batch 1_RIXI.fasta ./af2_results
source ~/colabfold_env/bin/activate
pip uninstall jax jaxlib -y
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install --force-reinstall dm-haiku
exit
python -c "import regex; print(regex.__file__)"
source .venv_clean/activate
source boltz/.venv_clean/activate
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=boltz_prediction
module purge
module load Python/3.11.3
source ~/boltz/.venv_clean/bin/activate
export PYTHONPATH=""
source .venv_clean/bin/activate
source .venv_clean/bin/activate
salloc --gpus=1 --mem=32G --time=02:00:00
salloc --gpus=1 --mem=64G --time=03:00:00 -C a100
salloc --gpus=v100:1 --mem=64G --time=03:00:00
exit
exit
# 1. Clean the slate
module purge
module load Python/3.11.3
export PYTHONPATH=""
# 2. Create the environment
python -m venv ~/bioemu_env
source ~/bioemu_env/bin/activate
# 3. Install BioEmu
pip install bioemu
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"
bioemu-sample --fasta 1_RIXI.fasta --out_dir ./bioemu_results --num_samples 10
source ~/bioemu_env/bin/activate
~/bioemu_env/bin/bioemu-sample --fasta 1_RIXI.fasta --out_dir ./bioemu_results --num_samples 10
~/bioemu_env/bin/bioemu-sample --fasta bioemu_env/1_RIXI.fasta --out_dir ./bioemu_results --num_samples 10
pip show bioemu
python -m bioemu.cli.sample --fasta 1_RIXI.fasta --out_dir ./bioemu_results --num_samples 10
# Ensure you are still in your environment
source ~/bioemu_env/bin/activate
# Use the full path to the script
~/bioemu_env/bin/bioemu-sample --fasta 1_RIXI.fasta --out_dir ./bioemu_results --num_samples 10
python -m bioemu.cli.sample --fasta 1_RIXI.fasta --out_dir ./bioemu_results --num_samples 10
python -m bioemu.sample --sequence 1_RIXI.fasta --num_samples 10 --output_dir ./bioemu_results
# Explicitly use the --fasta flag instead of --sequence
python -m bioemu.sample --fasta 1_RIXI.fasta --num_samples 10 --output_dir ./bioemu_results
python -m bioemu.sample 1_RIXI.fasta 10 ./bioemu_results
python -m bioemu.sample "AGKTGQMTVFWGRNKNEGTLKETCDTGLYTTVVISFYSVFGHGRYWGDLSGHDLRVIGADIKHCQSKNIFVFLSIGGAGKDYSLPTSKSAADVADNIWNAHMDGRRPGVFRPFGDAAVDGIDFFIDQGAPDHYDDLARNLYAYNKMYRARTPVRLTATVRCAFPDPRMKKALDTKLFERIHVRFYDDATCSYNHAGLAGVMAQWNKWTARYPGSHVYLGLAAANVPGKNDNVFIKQLYYDLLPNVQKAKNYGGIMLWDRFYDKQTGYGKTVKYWA" 10 ./bioemu_results
python -m bioemu.sample "ELCNPQDKQALLQIKKDLGNPTTLSSWLPTTDCCNRTWLGVLCDTDTQTYRVNNLDLSGLNLPKPYPIPSSLANLPYLNFLYIGGINNLVGPIPPAIAKLTQLHYLYITHTNVSGAIPDFLSQIKTLVTLDFSYNALSGTLPPSISSLPNLVGITFDGNRISGAIPDSYGSFSKLFTSMTISRNRLTGKIPPTFANLNLAFVDLSRNMLEGDASVLFGSDKNTQKIHLAKNSLAFDLGKVGLSKNLNGLDLRNNRIYGTLPQGLTQLKFLHSLDVSFNNLCGEIPQGGNLQRFDVSAYANNKCLCGSPLPACT" 10 ./bioemu_results
python -m bioemu.sample "ELCNPQDKQALLQIKKDLGNPTTLSSWLPTTDCCNRTWLGVLCDTDTQTYRVNNLDLSGLNLPKPYPIPSSLANLPYLNFLYIGGINNLVGPIPPAIAKLTQLHYLYITHTNVSGAIPDFLSQIKTLVTLDFSYNALSGTLPPSISSLPNLVGITFDGNRISGAIPDSYGSFSKLFTSMTISRNRLTGKIPPTFANLNLAFVDLSRNMLEGDASVLFGSDKNTQKIHLAKNSLAFDLGKVGLSKNLNGLDLRNNRIYGTLPQGLTQLKFLHSLDVSFNNLCGEIPQGGNLQRFDVSAYANNKCLCGSPLPACT" 10 ./bioemu_results/8IKW
python -m bioemu.sample "AGKTGQMTVFWGRNKNEGTLKETCDTGLYTTVVISFYSVFGHGRYWGDLSGHDLRVIGADIKHCQSKNIFVFLSIGGAGKDYSLPTSKSAADVADNIWNAHMDGRRPGVFRPFGDAAVDGIDFFIDQGAPDHYDDLARNLYAYNKMYRARTPVRLTATVRCAFPDPRMKKALDTKLFERIHVRFYDDATCSYNHAGLAGVMAQWNKWTARYPGSHVYLGLAAANVPGKNDNVFIKQLYYDLLPNVQKAKNYGGIMLWDRFYDKQTGYGKTVKYWA" 10 ./bioemu_results/RIXI
python -m bioemu.sample "MDQNQHLNKTAEAQPSENKKTRYCNGLKMFLAALSLSFIAKTLGAIIMKSSIIHIERRFEISSSLVGFIDGSFEIGNLLVIVFVSYFGSKLHRPKLIGIGCFIMGIGGVLTALPHFFMGYYRYSKETNINSSENSTSTLSTCLINQILSLNRASPEIVGKGCLKESGSYMWIYVFMGNMLRGIGETPIVPLGLSYIDDFAKEGHSSLYLGILNAIAMIGPIIGFTLGSLFSKMYVDIGYVDLSTIRITPTDSRWVGAWWLNFLVSGLFSIISSIPFFFLPQTPNKPQKERKASLSLHVLETNDEKDQTANLTNQGKNITKNVTGFFQSFKSILTNPLYVMFVLLTLLQVSSYIGAFTYVFKYVEQQYGQPSSKANILLGVITIPIFASGMFLGGYIIKKFKLNTVGIAKFSCFTAVMSLSFYLLYFFILCENKSVAGLTMTYDGNNPVTSHRDVPLSYCNSDCNCDESQWEPVCGNNGITYISPCLAGCKSSSGNKKPIVFYNCSCLEVTGLQNRNYSAHLGECPRDDACTRKFYFFVAIQVLNLFFSALGGTSHVMLIVKIVQPELKSLALGFHSMVIRALGGILAPIYFGALIDTTCIKWSTNNCGTRGSCRTYNSTSFSRVYLGLSSMLRVSSLVLYIILIYAMKKKYQEKDINASENGSVMDEANLESLNKNKHFVPSAGADSETHCLEGSDEVDA" 10 ./bioemu_results/8HND
python -m bioemu.sample "QSFEPEIEALKSFKNGISNDPLGVLSDWTIIGSLRHCNWTGITCDSTGHVVSVSLLEKQLEGVLSPAIANLTYLQVLDLTSNSFTGKIPAEIGKLTELQSLVLTENLLEGDIPAEIGNCSSLVQLELYDNQLTGKIPAELGNLVQLQALRIYKNKLTSSIPSSLFRLTQLTHLGLSENHLVGPISEEIGFLESLEVLTLHSNNFTGEFPQSITNLRNLTVLTVGFNNISGELPADLGLLTNLRNLSAHDNLLTGPIPSSISNCTGLKLLDLSHNQMTGEIPRGFGRMNLTFISIGRNHFTGEIPDDIFNCSNLETLSVADNNLTGTLKPLIGKLQKLRILQVSYNSLTGPIPREIGNLKDLNILYLHSNGFTGRIPREMSNLTLLQGLRMYSNDLEGPIPEEMFDMKLLSVLDLSNNKFSGQIPALFSKLESLTYLSLQGNKFNGSIPASLKSLSLLNTFDISDNLLTGTIPGELLASLKNMQLYLNFSNNLLTGTIPKELGKLEMVQEIDLSNNLFSGSIPRSLQACKNVFTLDFSQNNLSGHIPDEVFQGMDMIISLNLSRNSFSGEIPQSFGNMTHLVSLDLSSNNLTGEIPESLANLSTLKHLKLASNNLKGHVPESGVFKNINASDLMGNTDLCGSKKPLKPCTIKQKS" 10 ./bioemu_results/4MNA
sbatch run_bioemu.sb
ls run_bioemu.sb
ls bioemu_env/run_bioemu.sb
sbatch bioemu_env/run_bioemu.sb
git clone https://github.com/giacomo-janson/sam2.git
module purge
module load Python/3.11.3
export PYTHONPATH=""
# 2. Create the environment
python -m venv ~/asam_atomistic_env
source ~/asam_atomistic_env/bin/activate
# 3. Install core dependencies
pip install torch MDAnalysis numpy matplotlib tqdm
module purge
module load Python/3.11.3
export PYTHONPATH=""
source ~/asam_atomistic_env/bin/activate
python -m aSAM.train --data bioemu_results/RIXI/batch_0000000_0000001.npz --epochs 100 --out_dir ./asam_output
# Activate your aSAM environment
source ~/asam_atomistic_env/bin/activate
# Convert CIF to PDB
python -c "import MDAnalysis as mda; u = mda.Universe('boltz/boltz_results_1_RIXI/predictions/1_RIXI/1_RIXI_model_0.cif'); u.atoms.write('boltz/boltz_results_1_RIXI/predictions/1_RIXI/1_RIXI_model_0.pdb')"
# Activate your boltz environment
source ~/.venv_clean/bin/activate
# Run the conversion
python -c "import biotite.structure.io.pdb as pdb; import biotite.structure.io.pdbx as pdbx; cif = pdbx.PDBxFile.read('boltz/boltz_results_1_RIXI/predictions/1_RIXI/1_RIXI_model_0.cif'); struct = pdbx.get_structure(cif, model=1); out = pdb.PDBFile(); out.set_structure(struct); out.write('boltz/boltz_results_1_RIXI/predictions/1_RIXI/1_RIXI_model_0.pdb')"
pip install gemmi
gemmi cif2pdb boltz/boltz_results_1_RIXI/predictions/1_RIXI/1_RIXI_model_0.cif boltz/boltz_results_1_RIXI/predictions/1_RIXI/1_RIXI_model_0.pdb
gemmi help
gemmi --help
gemmi -help
python -c "import MDAnalysis as mda; u = mda.Universe('boltz/boltz_results_1_RIXI/predictions/1_RIXI/1_RIXI_model_0.cif'); u.atoms.write('boltz/boltz_results_1_RIXI/predictions/1_RIXI/1_RIXI_model_0.pdb')"
exit
boltz predict boltz/1_RIXI.yaml --use_msa_server --format cif,pdb
source .venv_clean/bin/activate
source boltz/.venv_clean/bin/activate
boltz predict boltz/1_RIXI.yaml --use_msa_server --format cif,pdb
boltz predict boltz/1_RIXI.yaml --use_msa_server --format pdb
boltz predict boltz/1_RIXI.yaml --use_msa_server --output_format cif,pdb
boltz predict boltz/1_RIXI.yaml --use_msa_server --output_format pdb
exit
python -c "import regex; print(regex.__file__)"
salloc --gpus=v100:1 --mem=64G --time=03:00:00
salloc --gpus=v100:1 --mem=64G --time=03:00:00
source /mnt/home/behren23/.bioemu_colabfold/bin/activate
exit
salloc --gpus=v100:1 --mem=64G --time=03:00:00
source /mnt/home/behren23/.bioemu_colabfold/bin/activate
salloc --gpus=v100:1 --mem=64G --time=01:00:00
source /mnt/home/behren23/.bioemu_colabfold/bin/activate
salloc --gpus=v100:1 --mem=64G --time=03:00:00
Proceed
cd ~/DPFunc
source dpfunc_env/bin/activate
jupyter notebook DataProcess/Process_data.ipynb
python -m jupyter notebook DataProcess/Process_data.ipynb
nvidia-smi
nvidia-smi
salloc --gpus=v100:1 --mem=64G --time=03:00:00
source .venv_clean/bin/activate
source boltz/.venv_clean/bin/activate
# Load environment
source /etc/profile.d/modules.sh
module load Python/3.11.3
cd ~/boltz && source .venv_clean/bin/activate
# Run Boltz on all 50 ligand files
for yaml in "ligand yaml files"/*.yaml; do     echo "Processing: $yaml";     boltz predict "$yaml" --use_msa_server --output_format pdb; done
exit
source /mnt/home/behren23/.bioemu_colabfold/bin/activate
salloc --gpus=v100:1 --mem=64G --time=03:00:00
salloc --gpus=v100:1 --mem=64G --time=03:00:00
python fix_scaler.py
python fix_scaler.py
cd /mnt/home/behren23
git remote add origin https://github.com/Behren23/CHE891_bioengineering_classwork.git
git branch -M main
git push -u origin main
git push -u origin main
git remote set-url origin https://github.com/Behren23/CHE891_bioengineering_classwork.git
git push -u origin main
git push origin main
git fetch origin
git diff origin/main
git pull origin main
git push origin main
cd /mnt/home/behren23
git remote add origin https://github.com/Behren23/CHE891_bioengineering_classwork.git
git branch -M main
git push -u origin main
git fetch origin
git diff origin/main
cd /mnt/home/behren23
git remote add origin https://github.com/Behren23/CHE891_bioengineering_classwork.git
git branch -M main
git push -u origin main
git pull origin main
cd /mnt/home/behren23
git remote add origin https://github.com/Behren23/CHE891_bioengineering_classwork.git
git branch -M main
git push -u origin main
git remote set-url origin https://github.com/Behren23/CHE891_bioengineering_classwork.git
git push -u origin main
cd /mnt/home/behren23
git remote add origin https://github.com/Behren23/CHE891_bioengineering_classwork.git
git branch -M main
git push -u origin main
git status
git add .
git status
git add onionnet/
git status
git add .
git status
git add sampling_so3_cache/
git status
git add .
git status
git add .
git add .
git status
git add sam2/
git add .
git status
git add .
git status
git add Documents/
git status
git add .
du -sh ~/*
du -sh ~/*
git add .
git commit -m "Add missing files"
git push -u origin main
git status
git commit -m "Add missing files"
git push -u origin main
git commit
