import sys
import numpy as np
import py3Dmol

from math import pow
from pathlib import Path
from io import StringIO
from Bio.PDB import PDBIO
from Bio.PDB import MMCIFParser, Superimposer
sys.path.append(str(Path("./src/simplefold").resolve()))


# following are example amino acid sequences:
example_sequences = {
    "7ftv_A": "GASKLRAVLEKLKLSRDDISTAAGMVKGVVDHLLLRLKCDSAFRGVGLLNTGSYYEHVKISAPNEFDVMFKLEVPRIQLEEYSNTRAYYFVKFKRNPKENPLSQFLEGEILSASKMLSKFRKIIKEEINDDTDVIMKRKRGGSPAVTLLISEKISVDITLALESKSSWPASTQEGLRIQNWLSAKVRKQLRLKPFYLVPKHAEETWRLSFSHIEKEILNNHGKSKTCCENKEEKCCRKDCLKLMKYLLEQLKERFKDKKHLDKFSSYHVKTAFFHVCTQNPQDSQWDRKDLGLCFDNCVTYFLQCLRTEKLENYFIPEFNLFSSNLIDKRSKEFLTKQIEYERNNEFPVFD",
    "8cny_A": "MGPSLDFALSLLRRNIRQVQTDQGHFTMLGVRDRLAVLPRHSQPGKTIWVEHKLINILDAVELVDEQGVNLELTLVTLDTNEKFRDITKFIPENISAASDATLVINTEHMPSMFVPVGDVVQYGFLNLSGKPTHRTMMYNFPTKAGQCGGVVTSVGKVIGIHIGGNGRQGFCAGLKRSYFAS",
    "8g8r_A": "GTVNWSVEDIVKGINSNNLESQLQATQAARKLLSREKQPPIDNIIRAGLIPKFVSFLGKTDCSPIQFESAWALTNIASGTSEQTKAVVDGGAIPAFISLLASPHAHISEQAVWALGNIAGDGSAFRDLVIKHGAIDPLLALLAVPDLSTLACGYLRNLTWTLSNLCRNKNPAPPLDAVEQILPTLVRLLHHNDPEVLADSCWAISYLTDGPNERIEMVVKKGVVPQLVKLLGATELPIVTPALRAIGNIVTGTDEQTQKVIDAGALAVFPSLLTNPKTNIQKEATWTMSNITAGRQDQIQQVVNHGLVPFLVGVLSKADFKTQKEAAWAITNYTSGGTVEQIVYLVHCGIIEPLMNLLSAKDTKIIQVILDAISNIFQAAEKLGETEKLSIMIEECGGLDKIEALQRHENESVYKASLNLIEKYFS",
    "8i85_A": "MGILQANRVLLSRLLPGVEPEGLTVRHGQFHQVVIASDRVVCLPRTAAAAARLPRRAAVMRVLAGLDLGCRTPRPLCEGSLPFLVLSRVPGAPLEADALEDSKVAEVVAAQYVTLLSGLASAGADEKVRAALPAPQGRWRQFAADVRAELFPLMSDGGCRQAERELAALDSLPDITEAVVHGNLGAENVLWVRDDGLPRLSGVIDWDEVSIGDPAEDLAAIGAGYGKDFLDQVLTLGGWSDRRMATRIATIRATFALQQALSACRDGDEEELADGLTGYR",
    "8g8r_A_x": "GTVNWSVEDIVKGINSNNLESQLQATQAARKLLSREKQPPIDNIIRAGLIPKFVSFLGKTDCSPIQFESAWALTNIASGTSEQTKAVVDGGAIPAFISLLASPHAHISEQAVWALGNIAGDGSAFRDLVIKHGAIDPLLALLAVPDLSTLACGYLRNLTWTLSNLCRNKNPAPPLDAVEQILPTLVRLLHHNDPEVLADSCWAISYLTDGPNERIEMVVKKGVVPQLVKLLGATELPIVTPALRAIGNIVTGTDEQTQKVIDAGALAVFPSLLTNPKTNIQKEATWTMSNITAGRQDQIQQVVNHGLVPFLVGVLSKADFKTQKEAAWAITNYTSGGTVEQIVYLVHCGIIEPLMNLLSAKDTKIIQVILDAISNIFQAAEKLGETEKLSIMIEECGGLDKIEALQRHENESVYKASLNLIEKYFSGTVNWSVEDIVKGINSNNLESQLQATQAARKLLSREKQPPIDNIIRAGLIPKFVSFLGKTDCSPIQFESAWALTNIASGTSEQTKAVVDGGAIPAFISLLASPHAHISEQAVWALGNIAGDGSAFRDLVIKHGAIDPLLALLAVPDLSTLACGYLRNLTWTLSNLCRNKNPAPPLDAVEQILPTLVRLLHHNDPEVLADSCWAISYLTDGPNERIEMVVKKGVVPQLVKLLGATELPIVTPALRAIGNIVTGTDEQTQKVIDAGALAVFPSLLTNPKTNIQKEATWTMSNITAGRQDQIQQVVNHGLVPFLVGVLSKADFKTQKEAAWAITNYTSGGTVEQIVYLVHCGIIEPLMNLLSAKDTKIIQVILDAISNIFQAAEKLGETEKLSIMIEECGGLDKIEALQRHENESVYKASLNLIEKYFSISEQAVWALGNIAGDGSAFRDLVIKHGAIDPLLALLAVPDLSTLACGYLRNLTWTLSNLCRNKNPAPPLDAVEQILPTLVRLLHHNDPEVLADSCWAISYLTDGPNERIEMVVKKGVVPQLVKLLGATELPIVTPALRAIGNIVTGTDEQTQKVIDAGALAVFPSLLTNPKTNIQKEATWTMSNITAGRQDQIQQVVNHGLVPFLVGVLSKADFKTQKEAAWAITNYTSGGTVEQIVYLVHCGIIEPLMNLLSAKDTKIIQVILDAISNIFQAAEKLGETEKLSIMIEECGGLDKIEALQRHENESVYKASLNLIEKYFSGTVNWSVEDIVKGINSNNLESQLQATQAARKLLSREKQPPIDNIIRAGLIPKFVSFLGKTDCSPIQFESAWALTNIASGTSEQTKAVVDGGAIPAFISLLASPHAHISEQAVWALGNIAGDGSAFRDLVIKHGAIDPLLALLAVPDLSTLACGYLRNLTWTLSNLCRNKNPAPPLDAVEQILPTLVRLLHHNDPEVLADSCWAISYLTDGPNERIEMVVKKGVVPQLVKLLGATELPIVTPALRAIGNIVTGTDEQTQKVIDAGALAVFPSLLTNPKTNIQKEATWTMSNITAGRQDQIQQVVNHGLVPFLVGVLSKADFKTQKEAAWAITNYTSGGTVEQIVYLVHCGIIEPLMNLLSAKDTKIIQVILDAISNIFQAAEKLGETEKLSIMIEECGGLDKIEALQRHENESVYKASLNLIEKYFS",
}
seq_id = "7ftv_A"  # choose from example_sequences
aa_sequence = example_sequences[seq_id][:32]
print(f"Predicting structure for {seq_id} with {len(aa_sequence)} amino acids.")


simplefold_model = "simplefold_100M" # choose from 100M, 360M, 700M, 1.1B, 1.6B, 3B
backend = "torch" # choose from ["mlx", "torch"]
ckpt_dir = "artifacts"
output_dir = "artifacts"
prediction_dir = f"predictions_{simplefold_model}_{backend}"
output_name = f"{seq_id}"
num_steps = 500 # number of inference steps for flow-matching
tau = 0.05 # stochasticity scale
plddt = True # whether to use pLDDT confidence module
nsample_per_protein = 1 # number of samples per protein


from src.simplefold.wrapper import ModelWrapper, InferenceWrapper

# initialize the folding model and pLDDT model
model_wrapper = ModelWrapper(
    simplefold_model=simplefold_model,
    ckpt_dir=ckpt_dir,
    plddt=plddt,
    backend=backend,
)
device = model_wrapper.device
folding_model = model_wrapper.from_pretrained_folding_model()
plddt_model = model_wrapper.from_pretrained_plddt_model()


# initialize the inference module with inference configurations
inference_wrapper = InferenceWrapper(
    output_dir=output_dir,
    prediction_dir=prediction_dir,
    num_steps=num_steps,
    tau=tau,
    nsample_per_protein=nsample_per_protein,
    device=device,
    backend=backend
)


# process input sequence and run inference
batch, structure, record = inference_wrapper.process_input(aa_sequence)
results = inference_wrapper.run_inference(
    batch,
    folding_model,
    plddt_model,
    device=device,
)
save_paths = inference_wrapper.save_result(
    structure,
    record,
    results,
    out_name=output_name
)

# visualize the first predicted structure
pdb_vis_dir = Path(output_dir) / prediction_dir
pdb_vis_dir.mkdir(parents=True, exist_ok=True)
pdb_path = save_paths[0]
view = py3Dmol.view(query=pdb_path)

# color based on the predicted confidence
# confidence coloring from low to high: red–orange–yellow–green–blue (0 to 100)
if plddt:
    view.setStyle({'cartoon':{'colorscheme':{'prop':'b','gradient':'roygb','min':0,'max':100}}})
else:
    view.setStyle({'cartoon':{'color':'spectrum'}})
view.zoomTo()

# save interactive HTML instead of showing inline
predicted_html_path = pdb_vis_dir / f"{output_name}_predicted.html"
with open(predicted_html_path, 'w', encoding='utf-8') as f:
    f.write(view._make_html())
print(f"Saved visualization: {predicted_html_path}")


# visualize the predicted structure in 3D alongside the GT structure

def calculate_tm_score(coords1, coords2, L_target=None):
    """
    Compute TM-score for two aligned coordinate sets (numpy arrays).
    
    coords1, coords2: Nx3 numpy arrays (aligned atomic coordinates, e.g. CA atoms)
    L_target: length of target protein (default = len(coords1))
    """
    assert coords1.shape == coords2.shape, "Aligned coords must have same shape"
    N = coords1.shape[0]

    if L_target is None:
        L_target = N

    # distances between aligned atoms
    dists = np.linalg.norm(coords1 - coords2, axis=1)

    # scaling factor d0
    d0 = 1.24 * pow(L_target - 15, 1/3) - 1.8
    if d0 < 0.5:
        d0 = 0.5  # safeguard, as in TM-align

    # TM-score
    score = np.sum(1.0 / (1.0 + (dists/d0)**2)) / L_target
    return score

parser = MMCIFParser(QUIET=True)

# Load two structures
struct1 = parser.get_structure("ref", f"assets/{seq_id}.cif")
struct2 = parser.get_structure("prd", pdb_path)

# Select CA atoms for alignment
atoms1 = [a for a in struct1.get_atoms() if a.get_id() == 'CA']
atoms2 = [a for a in struct2.get_atoms() if a.get_id() == 'CA']
print(len(atoms1), len(atoms2))

# Superimpose
sup = Superimposer()
sup.set_atoms(atoms1, atoms2)
sup.apply(struct2.get_atoms())

# Calculate TM-score
coords1 = np.array([a.coord for a in atoms1])
coords2 = np.array([a.coord for a in atoms2])
tm_score = calculate_tm_score(coords1, coords2)

print("TM-score (0-1, higher is better): {:.3f}".format(tm_score))
print("RMSD (lower is better): {:.3f}".format(sup.rms))

# Save aligned structures to strings
io = PDBIO()

s1_buf, s2_buf = StringIO(), StringIO()
io.set_structure(struct1); io.save(s1_buf)
io.set_structure(struct2); io.save(s2_buf)

# Visualize in py3Dmol
view = py3Dmol.view(width=600, height=400)
view.addModel(s1_buf.getvalue(),"pdb")
view.addModel(s2_buf.getvalue(),"pdb")

# Color reference protein blue, predicted structure red
view.setStyle({'model': 0}, {'cartoon': {'color': 'blue'}})
view.setStyle({'model': 1}, {'cartoon': {'color': 'red'}})

# Add legend
view.addLabel("Ground Truth", {'position': {'x': 0, 'y': 0, 'z': 0}, 'backgroundColor': 'blue', 'fontColor': 'white', 'fontSize': 12})
view.addLabel("Predicted", {'position': {'x': 0, 'y': 4, 'z': 0}, 'backgroundColor': 'red', 'fontColor': 'white', 'fontSize': 12})

view.zoomTo()

# save overlay visualization as interactive HTML
overlay_html_path = pdb_vis_dir / f"{output_name}_overlay.html"
with open(overlay_html_path, 'w', encoding='utf-8') as f:
    f.write(view._make_html())
print(f"Saved visualization: {overlay_html_path}")