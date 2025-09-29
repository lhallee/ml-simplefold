import sys
import numpy as np
import py3Dmol

from math import pow
from pathlib import Path
from io import StringIO
from Bio.PDB import PDBIO
from Bio.PDB import MMCIFParser, PDBParser, Superimposer
sys.path.append(str(Path("./src/simplefold").resolve()))


# utility to save py3Dmol views as PNG when possible, else HTML
def save_view_png_or_html(view, base_path: Path):
    png_path = base_path.with_suffix(".png")
    html_path = base_path.with_suffix(".html")
    try:
        png_bytes = view.png()
        with open(png_path, 'wb') as f:
            f.write(png_bytes)
        print(f"Saved visualization: {png_path}")
    except Exception as e:
        # fallback to HTML if PNG export is not supported in this environment
        try:
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(view._make_html())
            print(f"Saved visualization (HTML fallback): {html_path}")
        except Exception as e2:
            print(f"Failed to save visualization as PNG and HTML: {e2}")

# following are example amino acid sequences:
example_sequences = {
    "7ftv_A": "GASKLRAVLEKLKLSRDDISTAAGMVKGVVDHLLLRLKCDSAFRGVGLLNTGSYYEHVKISAPNEFDVMFKLEVPRIQLEEYSNTRAYYFVKFKRNPKENPLSQFLEGEILSASKMLSKFRKIIKEEINDDTDVIMKRKRGGSPAVTLLISEKISVDITLALESKSSWPASTQEGLRIQNWLSAKVRKQLRLKPFYLVPKHAEETWRLSFSHIEKEILNNHGKSKTCCENKEEKCCRKDCLKLMKYLLEQLKERFKDKKHLDKFSSYHVKTAFFHVCTQNPQDSQWDRKDLGLCFDNCVTYFLQCLRTEKLENYFIPEFNLFSSNLIDKRSKEFLTKQIEYERNNEFPVFD",
    "8cny_A": "MGPSLDFALSLLRRNIRQVQTDQGHFTMLGVRDRLAVLPRHSQPGKTIWVEHKLINILDAVELVDEQGVNLELTLVTLDTNEKFRDITKFIPENISAASDATLVINTEHMPSMFVPVGDVVQYGFLNLSGKPTHRTMMYNFPTKAGQCGGVVTSVGKVIGIHIGGNGRQGFCAGLKRSYFAS",
    "8g8r_A": "GTVNWSVEDIVKGINSNNLESQLQATQAARKLLSREKQPPIDNIIRAGLIPKFVSFLGKTDCSPIQFESAWALTNIASGTSEQTKAVVDGGAIPAFISLLASPHAHISEQAVWALGNIAGDGSAFRDLVIKHGAIDPLLALLAVPDLSTLACGYLRNLTWTLSNLCRNKNPAPPLDAVEQILPTLVRLLHHNDPEVLADSCWAISYLTDGPNERIEMVVKKGVVPQLVKLLGATELPIVTPALRAIGNIVTGTDEQTQKVIDAGALAVFPSLLTNPKTNIQKEATWTMSNITAGRQDQIQQVVNHGLVPFLVGVLSKADFKTQKEAAWAITNYTSGGTVEQIVYLVHCGIIEPLMNLLSAKDTKIIQVILDAISNIFQAAEKLGETEKLSIMIEECGGLDKIEALQRHENESVYKASLNLIEKYFS",
    "8i85_A": "MGILQANRVLLSRLLPGVEPEGLTVRHGQFHQVVIASDRVVCLPRTAAAAARLPRRAAVMRVLAGLDLGCRTPRPLCEGSLPFLVLSRVPGAPLEADALEDSKVAEVVAAQYVTLLSGLASAGADEKVRAALPAPQGRWRQFAADVRAELFPLMSDGGCRQAERELAALDSLPDITEAVVHGNLGAENVLWVRDDGLPRLSGVIDWDEVSIGDPAEDLAAIGAGYGKDFLDQVLTLGGWSDRRMATRIATIRATFALQQALSACRDGDEEELADGLTGYR",
    "8g8r_A_x": "GTVNWSVEDIVKGINSNNLESQLQATQAARKLLSREKQPPIDNIIRAGLIPKFVSFLGKTDCSPIQFESAWALTNIASGTSEQTKAVVDGGAIPAFISLLASPHAHISEQAVWALGNIAGDGSAFRDLVIKHGAIDPLLALLAVPDLSTLACGYLRNLTWTLSNLCRNKNPAPPLDAVEQILPTLVRLLHHNDPEVLADSCWAISYLTDGPNERIEMVVKKGVVPQLVKLLGATELPIVTPALRAIGNIVTGTDEQTQKVIDAGALAVFPSLLTNPKTNIQKEATWTMSNITAGRQDQIQQVVNHGLVPFLVGVLSKADFKTQKEAAWAITNYTSGGTVEQIVYLVHCGIIEPLMNLLSAKDTKIIQVILDAISNIFQAAEKLGETEKLSIMIEECGGLDKIEALQRHENESVYKASLNLIEKYFSGTVNWSVEDIVKGINSNNLESQLQATQAARKLLSREKQPPIDNIIRAGLIPKFVSFLGKTDCSPIQFESAWALTNIASGTSEQTKAVVDGGAIPAFISLLASPHAHISEQAVWALGNIAGDGSAFRDLVIKHGAIDPLLALLAVPDLSTLACGYLRNLTWTLSNLCRNKNPAPPLDAVEQILPTLVRLLHHNDPEVLADSCWAISYLTDGPNERIEMVVKKGVVPQLVKLLGATELPIVTPALRAIGNIVTGTDEQTQKVIDAGALAVFPSLLTNPKTNIQKEATWTMSNITAGRQDQIQQVVNHGLVPFLVGVLSKADFKTQKEAAWAITNYTSGGTVEQIVYLVHCGIIEPLMNLLSAKDTKIIQVILDAISNIFQAAEKLGETEKLSIMIEECGGLDKIEALQRHENESVYKASLNLIEKYFSISEQAVWALGNIAGDGSAFRDLVIKHGAIDPLLALLAVPDLSTLACGYLRNLTWTLSNLCRNKNPAPPLDAVEQILPTLVRLLHHNDPEVLADSCWAISYLTDGPNERIEMVVKKGVVPQLVKLLGATELPIVTPALRAIGNIVTGTDEQTQKVIDAGALAVFPSLLTNPKTNIQKEATWTMSNITAGRQDQIQQVVNHGLVPFLVGVLSKADFKTQKEAAWAITNYTSGGTVEQIVYLVHCGIIEPLMNLLSAKDTKIIQVILDAISNIFQAAEKLGETEKLSIMIEECGGLDKIEALQRHENESVYKASLNLIEKYFSGTVNWSVEDIVKGINSNNLESQLQATQAARKLLSREKQPPIDNIIRAGLIPKFVSFLGKTDCSPIQFESAWALTNIASGTSEQTKAVVDGGAIPAFISLLASPHAHISEQAVWALGNIAGDGSAFRDLVIKHGAIDPLLALLAVPDLSTLACGYLRNLTWTLSNLCRNKNPAPPLDAVEQILPTLVRLLHHNDPEVLADSCWAISYLTDGPNERIEMVVKKGVVPQLVKLLGATELPIVTPALRAIGNIVTGTDEQTQKVIDAGALAVFPSLLTNPKTNIQKEATWTMSNITAGRQDQIQQVVNHGLVPFLVGVLSKADFKTQKEAAWAITNYTSGGTVEQIVYLVHCGIIEPLMNLLSAKDTKIIQVILDAISNIFQAAEKLGETEKLSIMIEECGGLDKIEALQRHENESVYKASLNLIEKYFS",
}
seq_id = "7ftv_A"  # choose from example_sequences
aa_sequence = example_sequences[seq_id]
print(f"Predicting structure for {seq_id} with {len(aa_sequence)} amino acids.")


simplefold_model = "simplefold_3B" # choose from 100M, 360M, 700M, 1.1B, 1.6B, 3B
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

# save PNG with HTML fallback
save_view_png_or_html(view, pdb_vis_dir / f"{output_name}_predicted")


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
pdb_parser = PDBParser(QUIET=True)

# Load two structures (mmCIF for reference, PDB for prediction)
struct1 = parser.get_structure("ref", f"assets/{seq_id}.cif")
struct2 = pdb_parser.get_structure("prd", pdb_path)

# Determine chain ID from seq_id like "7ftv_A" -> "A"
chain_id = seq_id.split('_')[1] if '_' in seq_id and len(seq_id.split('_')) > 1 else None

def map_ca_by_residue(structure, chain_id_filter=None):
    ca_map = {}
    for model in structure:
        for chain in model:
            if chain_id_filter is not None and chain.id != chain_id_filter:
                continue
            for residue in chain:
                hetflag, resseq, icode = residue.get_id()
                if hetflag == ' ' and 'CA' in residue:
                    ca_map[(resseq, icode)] = residue['CA']
        break  # only first model
    return ca_map

# Build CA maps and align by common residues
ca_map_ref = map_ca_by_residue(struct1, chain_id)
ca_map_prd = map_ca_by_residue(struct2, chain_id)
print(len(ca_map_ref), len(ca_map_prd))

common_keys = sorted(set(ca_map_ref.keys()) & set(ca_map_prd.keys()))
if len(common_keys) == 0:
    print("No overlapping CA residues; skipping overlay visualization.")
else:
    atoms1 = [ca_map_ref[k] for k in common_keys]
    atoms2 = [ca_map_prd[k] for k in common_keys]

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

    # save overlay PNG with HTML fallback
    save_view_png_or_html(view, pdb_vis_dir / f"{output_name}_overlay")