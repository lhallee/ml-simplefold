#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as T
from torch import Tensor, from_numpy
from torch.nn.functional import one_hot, pad
from einops import repeat, rearrange
from dataclasses import astuple, dataclass, asdict
from abc import ABC, abstractmethod
from typing import Optional
from pathlib import Path
from types import SimpleNamespace
from tqdm import tqdm
from transformers import AutoModel, PretrainedConfig, PreTrainedModel


ckpt_url_dict = {
    "simplefold_100M": "https://ml-site.cdn-apple.com/models/simplefold/simplefold_100M.ckpt",
    "simplefold_360M": "https://ml-site.cdn-apple.com/models/simplefold/simplefold_360M.ckpt",
    "simplefold_700M": "https://ml-site.cdn-apple.com/models/simplefold/simplefold_700M.ckpt",
    "simplefold_1.1B": "https://ml-site.cdn-apple.com/models/simplefold/simplefold_1.1B.ckpt",
    "simplefold_1.6B": "https://ml-site.cdn-apple.com/models/simplefold/simplefold_1.6B.ckpt",
    "simplefold_3B": "https://ml-site.cdn-apple.com/models/simplefold/simplefold_3B.ckpt",
}

plddt_ckpt_url = (
    "https://ml-site.cdn-apple.com/models/simplefold/plddt_module_1.6B.ckpt"
)

mx = None  # optional MLX placeholder


# Minimal residue and token constants required for direct-from-sequence tokenization
residue_constants = SimpleNamespace()
residue_constants.restypes = [
    "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
    "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",
]
residue_constants.restypes_with_x = list(residue_constants.restypes) + ["X"]
residue_constants.restype_order_with_x = {
    aa: i for i, aa in enumerate(residue_constants.restypes_with_x)
}

# Subset of boltz constants used in tokenization/featurization
class _Const:
    pass

const = _Const()

const.chain_types = ["PROTEIN", "DNA", "RNA", "NONPOLYMER"]
const.chain_type_ids = {c: i for i, c in enumerate(const.chain_types)}

const.tokens = [
    "<pad>", "-",
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "UNK",
    # nucleic placeholders kept for dtype compatibility
    "A", "G", "C", "U", "N", "DA", "DG", "DC", "DT", "DN",
]
const.token_ids = {t: i for i, t in enumerate(const.tokens)}
const.num_tokens = len(const.tokens)
const.unk_token = {"PROTEIN": "UNK", "DNA": "DN", "RNA": "N"}
const.num_elements = 128
const.pocket_contact_info = {"UNSPECIFIED": 0, "UNSELECTED": 1, "POCKET": 2, "BINDER": 3}

# Three-letter protein residue atom definitions (subset for proteins)
const.ref_atoms = {
    "PAD": [],
    "UNK": ["N", "CA", "C", "O", "CB"],
    "-": [],
    "ALA": ["N", "CA", "C", "O", "CB"],
    "ARG": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"],
    "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"],
    "CYS": ["N", "CA", "C", "O", "CB", "SG"],
    "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"],
    "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"],
    "GLY": ["N", "CA", "C", "O"],
    "HIS": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"],
    "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"],
    "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"],
    "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"],
    "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"],
    "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD"],
    "SER": ["N", "CA", "C", "O", "CB", "OG"],
    "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2"],
    "TRP": [
        "N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2",
        "CE3", "CZ2", "CZ3", "CH2",
    ],
    "TYR": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2"],
}

# Letter<->token helpers
prot_letter_to_token = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "E": "GLU", "Q": "GLN", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
    "X": "UNK", "B": "UNK", "Z": "UNK", "J": "UNK", "O": "UNK", "U": "UNK",
}


Atom = [
    ("name", np.dtype("4i1")),
    ("element", np.dtype("i1")),
    ("charge", np.dtype("i1")),
    ("coords", np.dtype("3f4")),
    ("conformer", np.dtype("3f4")),
    ("is_present", np.dtype("?")),
    ("chirality", np.dtype("i1")),
]
Bond = [
    ("atom_1", np.dtype("i4")),
    ("atom_2", np.dtype("i4")),
    ("type", np.dtype("i1")),
]
Residue = [
    ("name", np.dtype("<U5")),
    ("res_type", np.dtype("i1")),
    ("res_idx", np.dtype("i4")),
    ("atom_idx", np.dtype("i4")),
    ("atom_num", np.dtype("i4")),
    ("atom_center", np.dtype("i4")),
    ("atom_disto", np.dtype("i4")),
    ("is_standard", np.dtype("?")),
    ("is_present", np.dtype("?")),
]
Chain = [
    ("name", np.dtype("<U5")),
    ("mol_type", np.dtype("i1")),
    ("entity_id", np.dtype("i4")),
    ("sym_id", np.dtype("i4")),
    ("asym_id", np.dtype("i4")),
    ("atom_idx", np.dtype("i4")),
    ("atom_num", np.dtype("i4")),
    ("res_idx", np.dtype("i4")),
    ("res_num", np.dtype("i4")),
]
Connection = [
    ("chain_1", np.dtype("i4")),
    ("chain_2", np.dtype("i4")),
    ("res_1", np.dtype("i4")),
    ("res_2", np.dtype("i4")),
    ("atom_1", np.dtype("i4")),
    ("atom_2", np.dtype("i4")),
]
Interface = [
    ("chain_1", np.dtype("i4")),
    ("chain_2", np.dtype("i4")),
]
Token = [
    ("token_idx", np.dtype("i4")),
    ("atom_idx", np.dtype("i4")),
    ("atom_num", np.dtype("i4")),
    ("res_idx", np.dtype("i4")),
    ("res_type", np.dtype("i1")),
    ("sym_id", np.dtype("i4")),
    ("asym_id", np.dtype("i4")),
    ("entity_id", np.dtype("i4")),
    ("mol_type", np.dtype("i1")),
    ("center_idx", np.dtype("i4")),
    ("disto_idx", np.dtype("i4")),
    ("center_coords", np.dtype("3f4")),
    ("disto_coords", np.dtype("3f4")),
    ("resolved_mask", np.dtype("?")),
    ("disto_mask", np.dtype("?")),
]
TokenBond = [
    ("token_1", np.dtype("i4")),
    ("token_2", np.dtype("i4")),
]
MSAResidue = [
    ("res_type", np.dtype("i1")),
]
MSADeletion = [
    ("res_idx", np.dtype("i2")),
    ("deletion", np.dtype("i2")),
]
MSASequence = [
    ("seq_idx", np.dtype("i2")),
    ("taxonomy", np.dtype("i4")),
    ("res_start", np.dtype("i4")),
    ("res_end", np.dtype("i4")),
    ("del_start", np.dtype("i4")),
    ("del_end", np.dtype("i4")),
]


class NumpySerializable:
    """Serializable datatype."""

    @classmethod
    def load(cls: "NumpySerializable", path: Path) -> "NumpySerializable":
        """Load the object from an NPZ file.

        Parameters
        ----------
        path : Path
            The path to the file.

        Returns
        -------
        Serializable
            The loaded object.

        """
        return cls(**np.load(path))

    def dump(self, path: Path) -> None:
        """Dump the object to an NPZ file.

        Parameters
        ----------
        path : Path
            The path to the file.

        """
        np.savez_compressed(str(path), **asdict(self))


@dataclass(frozen=True)
class Structure(NumpySerializable):
    """Structure datatype."""

    atoms: np.ndarray
    bonds: np.ndarray
    residues: np.ndarray
    chains: np.ndarray
    connections: np.ndarray
    interfaces: np.ndarray
    mask: np.ndarray

    @classmethod
    def load(cls: "Structure", path: Path) -> "Structure":
        """Load a structure from an NPZ file.

        Parameters
        ----------
        path : Path
            The path to the file.

        Returns
        -------
        Structure
            The loaded structure.

        """
        structure = np.load(path)
        return cls(
            atoms=structure["atoms"],
            bonds=structure["bonds"],
            residues=structure["residues"],
            chains=structure["chains"],
            connections=structure["connections"].astype(Connection),
            interfaces=structure["interfaces"],
            mask=structure["mask"],
        )

    def remove_invalid_chains(self) -> "Structure":  # noqa: PLR0915
        """Remove invalid chains.

        Parameters
        ----------
        structure : Structure
            The structure to process.

        Returns
        -------
        Structure
            The structure with masked chains removed.

        """
        entity_counter = {}
        atom_idx, res_idx, chain_idx = 0, 0, 0
        atoms, residues, chains = [], [], []
        atom_map, res_map, chain_map = {}, {}, {}
        for i, chain in enumerate(self.chains):
            # Skip masked chains
            if not self.mask[i]:
                continue

            # Update entity counter
            entity_id = chain["entity_id"]
            if entity_id not in entity_counter:
                entity_counter[entity_id] = 0
            else:
                entity_counter[entity_id] += 1

            # Update the chain
            new_chain = chain.copy()
            new_chain["atom_idx"] = atom_idx
            new_chain["res_idx"] = res_idx
            new_chain["asym_id"] = chain_idx
            new_chain["sym_id"] = entity_counter[entity_id]
            chains.append(new_chain)
            chain_map[i] = chain_idx
            chain_idx += 1

            # Add the chain residues
            res_start = chain["res_idx"]
            res_end = chain["res_idx"] + chain["res_num"]
            for j, res in enumerate(self.residues[res_start:res_end]):
                # Update the residue
                new_res = res.copy()
                new_res["atom_idx"] = atom_idx
                new_res["atom_center"] = (
                    atom_idx + new_res["atom_center"] - res["atom_idx"]
                )
                new_res["atom_disto"] = (
                    atom_idx + new_res["atom_disto"] - res["atom_idx"]
                )
                residues.append(new_res)
                res_map[res_start + j] = res_idx
                res_idx += 1

                # Update the atoms
                start = res["atom_idx"]
                end = res["atom_idx"] + res["atom_num"]
                atoms.append(self.atoms[start:end])
                atom_map.update({k: atom_idx + k - start for k in range(start, end)})
                atom_idx += res["atom_num"]

        # Concatenate the tables
        atoms = np.concatenate(atoms, dtype=Atom)
        residues = np.array(residues, dtype=Residue)
        chains = np.array(chains, dtype=Chain)

        # Update bonds
        bonds = []
        for bond in self.bonds:
            atom_1 = bond["atom_1"]
            atom_2 = bond["atom_2"]
            if (atom_1 in atom_map) and (atom_2 in atom_map):
                new_bond = bond.copy()
                new_bond["atom_1"] = atom_map[atom_1]
                new_bond["atom_2"] = atom_map[atom_2]
                bonds.append(new_bond)

        # Update connections
        connections = []
        for connection in self.connections:
            chain_1 = connection["chain_1"]
            chain_2 = connection["chain_2"]
            res_1 = connection["res_1"]
            res_2 = connection["res_2"]
            atom_1 = connection["atom_1"]
            atom_2 = connection["atom_2"]
            if (atom_1 in atom_map) and (atom_2 in atom_map):
                new_connection = connection.copy()
                new_connection["chain_1"] = chain_map[chain_1]
                new_connection["chain_2"] = chain_map[chain_2]
                new_connection["res_1"] = res_map[res_1]
                new_connection["res_2"] = res_map[res_2]
                new_connection["atom_1"] = atom_map[atom_1]
                new_connection["atom_2"] = atom_map[atom_2]
                connections.append(new_connection)

        # Create arrays
        bonds = np.array(bonds, dtype=Bond)
        connections = np.array(connections, dtype=Connection)
        interfaces = np.array([], dtype=Interface)
        mask = np.ones(len(chains), dtype=bool)

        return Structure(
            atoms=atoms,
            bonds=bonds,
            residues=residues,
            chains=chains,
            connections=connections,
            interfaces=interfaces,
            mask=mask,
        )


@dataclass(frozen=True)
class MSA(NumpySerializable):
    """MSA datatype."""

    sequences: np.ndarray
    deletions: np.ndarray
    residues: np.ndarray


@dataclass(frozen=True)
class Tokenized:
    """Tokenized datatype."""

    tokens: np.ndarray
    bonds: np.ndarray
    structure: Structure
    msa: dict[str, MSA]


@dataclass
class Input:
    structure: Structure
    msa: dict


@dataclass
class TokenData:
    """TokenData datatype."""

    token_idx: int
    atom_idx: int
    atom_num: int
    res_idx: int
    res_type: int
    sym_id: int
    asym_id: int
    entity_id: int
    mol_type: int
    center_idx: int
    disto_idx: int
    center_coords: np.ndarray
    disto_coords: np.ndarray
    resolved_mask: bool
    disto_mask: bool


class Tokenizer(ABC):
    """Tokenize an input structure for training."""

    @abstractmethod
    def tokenize(self, data: Input) -> Tokenized:
        """Tokenize the input data.

        Parameters
        ----------
        data : Input
            The input data.

        Returns
        -------
        Tokenized
            The tokenized data.

        """
        raise NotImplementedError


class BoltzTokenizer(Tokenizer):
    """Tokenize an input structure for training."""

    def tokenize(self, data: Input) -> Tokenized:
        """Tokenize the input data.

        Parameters
        ----------
        data : Input
            The input data.

        Returns
        -------
        Tokenized
            The tokenized data.

        """
        # Get structure data
        struct = data.structure

        # Create token data
        token_data = []

        # Keep track of atom_idx to token_idx
        token_idx = 0
        atom_to_token = {}

        # Filter to valid chains only
        chains = struct.chains[struct.mask]

        for chain in chains:
            # Get residue indices
            res_start = chain["res_idx"]
            res_end = chain["res_idx"] + chain["res_num"]

            if chain["mol_type"] != const.chain_type_ids["PROTEIN"]:
                # Skip non-protein chains
                continue

            for res in struct.residues[res_start:res_end]:
                # Get atom indices
                atom_start = res["atom_idx"]
                atom_end = res["atom_idx"] + res["atom_num"]

                # Standard residues are tokens
                if res["is_standard"]:
                    # Get center and disto atoms
                    center = struct.atoms[res["atom_center"]]
                    disto = struct.atoms[res["atom_disto"]]

                    # Token is present if centers are
                    is_present = res["is_present"] & center["is_present"]
                    is_disto_present = res["is_present"] & disto["is_present"]

                    # Apply chain transformation
                    c_coords = center["coords"]
                    d_coords = disto["coords"]

                    # Create token
                    token = TokenData(
                        token_idx=token_idx,
                        atom_idx=res["atom_idx"],
                        atom_num=res["atom_num"],
                        res_idx=res["res_idx"],
                        res_type=res["res_type"],
                        sym_id=chain["sym_id"],
                        asym_id=chain["asym_id"],
                        entity_id=chain["entity_id"],
                        mol_type=chain["mol_type"],
                        center_idx=res["atom_center"],
                        disto_idx=res["atom_disto"],
                        center_coords=c_coords,
                        disto_coords=d_coords,
                        resolved_mask=is_present,
                        disto_mask=is_disto_present,
                    )
                    token_data.append(astuple(token))

                    # Update atom_idx to token_idx
                    for atom_idx in range(atom_start, atom_end):
                        atom_to_token[atom_idx] = token_idx

                    token_idx += 1

                # Non-standard are tokenized per atom
                else:
                    # We use the unk protein token as res_type
                    unk_token = const.unk_token["PROTEIN"]
                    unk_id = const.token_ids[unk_token]

                    # Get atom coordinates
                    atom_data = struct.atoms[atom_start:atom_end]
                    atom_coords = atom_data["coords"]

                    # Tokenize each atom
                    for i, atom in enumerate(atom_data):
                        # Token is present if atom is
                        is_present = res["is_present"] & atom["is_present"]
                        index = atom_start + i

                        # Create token
                        token = TokenData(
                            token_idx=token_idx,
                            atom_idx=index,
                            atom_num=1,
                            res_idx=res["res_idx"],
                            res_type=unk_id,
                            sym_id=chain["sym_id"],
                            asym_id=chain["asym_id"],
                            entity_id=chain["entity_id"],
                            mol_type=chain["mol_type"],
                            center_idx=index,
                            disto_idx=index,
                            center_coords=atom_coords[i],
                            disto_coords=atom_coords[i],
                            resolved_mask=is_present,
                            disto_mask=is_present,
                        )
                        token_data.append(astuple(token))

                        # Update atom_idx to token_idx
                        atom_to_token[index] = token_idx
                        token_idx += 1

        # Create token bonds
        token_bonds = []

        # Add atom-atom bonds from ligands
        for bond in struct.bonds:
            if (
                bond["atom_1"] not in atom_to_token
                or bond["atom_2"] not in atom_to_token
            ):
                continue
            token_bond = (
                atom_to_token[bond["atom_1"]],
                atom_to_token[bond["atom_2"]],
            )
            token_bonds.append(token_bond)

        # Add connection bonds (covalent)
        for conn in struct.connections:
            if (
                conn["atom_1"] not in atom_to_token
                or conn["atom_2"] not in atom_to_token
            ):
                continue
            token_bond = (
                atom_to_token[conn["atom_1"]],
                atom_to_token[conn["atom_2"]],
            )
            token_bonds.append(token_bond)

        token_data = np.array(token_data, dtype=Token)
        token_bonds = np.array(token_bonds, dtype=TokenBond)
        tokenized = Tokenized(
            token_data,
            token_bonds,
            data.structure,
            data.msa,
        )
        return tokenized


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def compute_aggregated_metric(logits, end=1.0):
    """Compute the metric from the logits.

    Parameters
    ----------
    logits : torch.Tensor
        The logits of the metric
    end : float
        Max value of the metric, by default 1.0

    Returns
    -------
    Tensor
        The metric value

    """
    num_bins = logits.shape[-1]
    bin_width = end / num_bins
    bounds = torch.arange(
        start=0.5 * bin_width, end=end, step=bin_width, device=logits.device
    )
    probs = nn.functional.softmax(logits, dim=-1)
    plddt = torch.sum(
        probs * bounds.view(*((1,) * len(probs.shape[:-1])), *bounds.shape),
        dim=-1,
    )
    return plddt


class AbsolutePositionEncoding(nn.Module):
    def __init__(self, in_dim, embed_dim, include_input=False):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = embed_dim
        self.include_input = include_input
        assert embed_dim % in_dim == 0, "embed_dim must be divisible by in_dim"
        self.embed_dim = embed_dim + in_dim if include_input else embed_dim

    def forward(self, pos):
        pos_embs = []
        for i in range(self.in_dim):
            pe = self.get_1d_pos_embed(pos[..., i])
            pos_embs.append(pe)
        if self.include_input:
            pos_embs.append(pos)
        pos_embs = torch.cat(pos_embs, dim=-1)
        return pos_embs

    def get_1d_pos_embed(self, pos):
        """
        https://github.com/facebookresearch/DiT/blob/main/models.py#L303
        """
        embed_dim = self.hidden_dim // (self.in_dim * 2)
        omega = 2 ** torch.linspace(0, math.log(224, 2) - 1, embed_dim).to(pos.device)
        omega *= torch.pi

        if len(pos.shape) == 1:
            out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product
        elif len(pos.shape) == 2:
            out = torch.einsum("nm,d->nmd", pos, omega)

        emb_sin = torch.sin(out)  # (*, M, D/2)
        emb_cos = torch.cos(out)  # (*, M, D/2)
        emb = torch.cat([emb_sin, emb_cos], dim=-1)  # (*, M, D)
        return emb


class FourierPositionEncoding(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        include_input: bool = False,
        min_freq_log2: float = 0,
        max_freq_log2: float = 12,
        num_freqs: int = 32,
        log_sampling: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.include_input = include_input
        self.min_freq_log2 = min_freq_log2
        self.max_freq_log2 = max_freq_log2
        self.num_freqs = num_freqs
        self.log_sampling = log_sampling
        self.create_embedding_fn()

    def create_embedding_fn(self):
        d = self.in_dim
        dim_out = 0
        if self.include_input:
            dim_out += d

        min_freq = self.min_freq_log2
        max_freq = self.max_freq_log2
        N_freqs = self.num_freqs

        if self.log_sampling:
            freq_bands = 2.0 ** torch.linspace(
                min_freq, max_freq, steps=N_freqs
            )  # (nf,)
        else:
            freq_bands = torch.linspace(
                2.0**min_freq, 2.0**max_freq, steps=N_freqs
            )  # (nf,)

        assert (
            freq_bands.isfinite().all()
        ), f"nan: {freq_bands.isnan().any()} inf: {freq_bands.isinf().any()}"

        self.register_buffer("freq_bands", freq_bands)  # (nf,)
        self.embed_dim = dim_out + d * self.freq_bands.numel() * 2

    def forward(
        self,
        pos: torch.Tensor,
    ):
        """
        Get the positional encoding for each coordinate.
        Args:
            pos:
                (*, in_dim)
        Returns:
            out:
                (*, in_dimitional_encoding)
        """

        out = []
        if self.include_input:
            out = [pos]  # (*, in_dim)

        pos = pos.unsqueeze(-1) * self.freq_bands  # (*b, d, nf)

        out += [
            torch.sin(pos).flatten(start_dim=-2),  # (*b, d*nf)
            torch.cos(pos).flatten(start_dim=-2),  # (*b, d*nf)
        ]

        out = torch.cat(out, dim=-1)  # (*b, 2 * in_dim * nf (+ in_dim))
        return out


def compute_axial_cis(
    ts: torch.Tensor,
    in_dim: int,
    dim: int,
    theta: float = 100.0,
):
    B, N, D = ts.shape
    freqs_all = []
    interval = 2 * in_dim
    for i in range(in_dim):
        freq = 1.0 / (
            theta ** (torch.arange(0, dim, interval)[: (dim // interval)].float() / dim)
        ).to(ts.device)
        t = ts[..., i].flatten()
        freq_i = torch.outer(t, freq)
        freq_cis_i = torch.polar(torch.ones_like(freq_i), freq_i)
        freq_cis_i = freq_cis_i.view(B, N, -1)
        freqs_all.append(freq_cis_i)
    freqs_cis = torch.cat(freqs_all, dim=-1)
    return freqs_cis


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


class AxialRotaryPositionEncoding(nn.Module):
    def __init__(
        self,
        in_dim,
        embed_dim,
        num_heads,
        base=100.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.num_heads = num_heads
        self.embed_dim = embed_dim // num_heads
        self.base = base

    def forward(self, xq, xk, pos):
        """
        xq: [B, H, N, D]
        xk: [B, H, N, D]
        pos: [B, N, in_dim]
        """
        if pos.ndim == 2:
            pos = pos.unsqueeze(-1)
        freqs_cis = compute_axial_cis(pos, self.in_dim, self.embed_dim, self.base)
        freqs_cis = freqs_cis.unsqueeze(1)
        return apply_rotary_emb(xq, xk, freqs_cis.to(xq.device))


def encode_sequence(
    seq: str,
    residue_index_offset: T.Optional[int] = 512,
    chain_linker: T.Optional[str] = "G" * 25,
) -> T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if chain_linker is None:
        chain_linker = ""
    if residue_index_offset is None:
        residue_index_offset = 0

    chains = seq.split(":")
    seq = chain_linker.join(chains)

    unk_idx = residue_constants.restype_order_with_x["X"]
    encoded = torch.tensor(
        [residue_constants.restype_order_with_x.get(aa, unk_idx) for aa in seq]
    )
    residx = torch.arange(len(encoded))

    if residue_index_offset > 0:
        start = 0
        for i, chain in enumerate(chains):
            residx[start : start + len(chain) + len(chain_linker)] += (
                i * residue_index_offset
            )
            start += len(chain) + len(chain_linker)

    linker_mask = torch.ones_like(encoded, dtype=torch.float32)
    chain_index = []
    offset = 0
    for i, chain in enumerate(chains):
        if i > 0:
            chain_index.extend([i - 1] * len(chain_linker))
        chain_index.extend([i] * len(chain))
        offset += len(chain)
        linker_mask[offset : offset + len(chain_linker)] = 0
        offset += len(chain_linker)

    chain_index = torch.tensor(chain_index, dtype=torch.int64)

    return encoded, residx, linker_mask, chain_index


def batch_encode_sequences(
    sequences: T.Sequence[str],
    residue_index_offset: T.Optional[int] = 512,
    chain_linker: T.Optional[str] = "G" * 25,
) -> T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    aatype_list = []
    residx_list = []
    linker_mask_list = []
    chain_index_list = []
    for seq in sequences:
        aatype_seq, residx_seq, linker_mask_seq, chain_index_seq = encode_sequence(
            seq,
            residue_index_offset=residue_index_offset,
            chain_linker=chain_linker,
        )
        aatype_list.append(aatype_seq)
        residx_list.append(residx_seq)
        linker_mask_list.append(linker_mask_seq)
        chain_index_list.append(chain_index_seq)

    aatype = collate_dense_tensors(aatype_list)
    mask = collate_dense_tensors(
        [aatype.new_ones(len(aatype_seq)) for aatype_seq in aatype_list]
    )
    residx = collate_dense_tensors(residx_list)
    linker_mask = collate_dense_tensors(linker_mask_list)
    chain_index_list = collate_dense_tensors(chain_index_list, -1)

    return aatype, mask, residx, linker_mask, chain_index_list


def collate_dense_tensors(
    samples: T.List[torch.Tensor], pad_v: float = 0
) -> torch.Tensor:
    """
    Takes a list of tensors with the following dimensions:
        [(d_11,       ...,           d_1K),
         (d_21,       ...,           d_2K),
         ...,
         (d_N1,       ...,           d_NK)]
    and stack + pads them into a single tensor of:
    (N, max_i=1,N { d_i1 }, ..., max_i=1,N {diK})
    """
    if len(samples) == 0:
        return torch.Tensor()
    if len(set(x.dim() for x in samples)) != 1:
        raise RuntimeError(
            f"Samples has varying dimensions: {[x.dim() for x in samples]}"
        )
    (device,) = tuple(set(x.device for x in samples))  # assumes all on same device
    max_shape = [max(lst) for lst in zip(*[x.shape for x in samples])]
    result = torch.empty(
        len(samples), *max_shape, dtype=samples[0].dtype, device=device
    )
    result.fill_(pad_v)
    for i in range(len(samples)):
        result_i = result[i]
        t = samples[i]
        result_i[tuple(slice(0, k) for k in t.shape)] = t
    return result


def _af2_to_esm(d):
    # Remember that t is shifted from residue_constants by 1 (0 is padding).
    esm_reorder = [d.padding_idx] + [
        d.get_idx(v) for v in residue_constants.restypes_with_x
    ]
    return torch.tensor(esm_reorder)


def af2_idx_to_esm_idx(aa, mask, af2_to_esm):
    aa = (aa + 1).masked_fill(mask != 1, 0)
    return af2_to_esm[aa]


def compute_language_model_representations(
    esmaa, esm, esm_dict, backend="torch"
) -> torch.Tensor:
    """Adds bos/eos tokens for the language model, since the structure module doesn't use these."""
    batch_size = esmaa.size(0)

    bosi, eosi = esm_dict.cls_idx, esm_dict.eos_idx
    bos = esmaa.new_full((batch_size, 1), bosi)
    eos = esmaa.new_full((batch_size, 1), esm_dict.padding_idx)
    esmaa = torch.cat([bos, esmaa, eos], dim=1)
    # Use the first padding index as eos during inference.
    esmaa[range(batch_size), (esmaa != 1).sum(1)] = eosi

    if backend == "mlx":
        esmaa = mx.array(esmaa)

    res = esm(
        esmaa,
        repr_layers=range(esm.num_layers + 1),
        need_head_weights=False,
    )
    if backend == "mlx":
        res['representations'] = {k: torch.from_numpy(np.array(v)) for k,v in res['representations'].items()}

    esm_s = torch.stack(
        [v for _, v in sorted(res["representations"].items())], dim=2
    )
    esm_s = esm_s[:, 1:-1]  # B, L, nLayers, C
    return esm_s, None


def pad_dim(data: Tensor, dim: int, pad_len: float, value: float = 0) -> Tensor:
    """Pad a tensor along a given dimension.

    Parameters
    ----------
    data : Tensor
        The input tensor.
    dim : int
        The dimension to pad.
    pad_len : float
        The padding length.
    value : int, optional
        The value to pad with.

    Returns
    -------
    Tensor
        The padded tensor.

    """
    if pad_len == 0:
        return data

    total_dims = len(data.shape)
    padding = [0] * (2 * (total_dims - dim))
    padding[2 * (total_dims - 1 - dim) + 1] = pad_len
    return pad(data, tuple(padding), value=value)


def select_subset_from_mask(mask, p):
    num_true = np.sum(mask)
    v = np.random.geometric(p) + 1
    k = min(v, num_true)

    true_indices = np.where(mask)[0]

    # Randomly select k indices from the true_indices
    selected_indices = np.random.choice(true_indices, size=k, replace=False)

    new_mask = np.zeros_like(mask)
    new_mask[selected_indices] = 1

    return new_mask


def process_token_features(
    data: Tokenized,
    max_tokens: Optional[int] = None,
) -> dict[str, Tensor]:
    """Get the token features.

    Parameters
    ----------
    data : Tokenized
        The tokenized data.
    max_tokens : int
        The maximum number of tokens.

    Returns
    -------
    dict[str, Tensor]
        The token features.

    """
    # Token data
    token_data = data.tokens

    # Token core features
    token_index = torch.arange(len(token_data), dtype=torch.long)
    residue_index = from_numpy(token_data["res_idx"]).long()
    asym_id = from_numpy(token_data["asym_id"]).long()
    entity_id = from_numpy(token_data["entity_id"]).long()
    sym_id = from_numpy(token_data["sym_id"]).long()
    mol_type = from_numpy(token_data["mol_type"]).long()
    res_type = from_numpy(token_data["res_type"]).long()
    res_type = one_hot(res_type, num_classes=const.num_tokens)
    disto_center = from_numpy(token_data["disto_coords"])

    # Token mask features
    pad_mask = torch.ones(len(token_data), dtype=torch.float)
    resolved_mask = from_numpy(token_data["resolved_mask"]).float()
    disto_mask = from_numpy(token_data["disto_mask"]).float()

    # Pocket conditioned feature 
    # (dummy feature in SimpleFold)
    pocket_feature = (
        np.zeros(len(token_data)) + const.pocket_contact_info["UNSPECIFIED"]
    )
    pocket_feature = from_numpy(pocket_feature).long()
    pocket_feature = one_hot(pocket_feature, num_classes=len(const.pocket_contact_info))

    # Pad to max tokens if given
    if max_tokens is not None:
        pad_len = max_tokens - len(token_data)
        if pad_len > 0:
            token_index = pad_dim(token_index, 0, pad_len)
            residue_index = pad_dim(residue_index, 0, pad_len)
            asym_id = pad_dim(asym_id, 0, pad_len)
            entity_id = pad_dim(entity_id, 0, pad_len)
            sym_id = pad_dim(sym_id, 0, pad_len)
            mol_type = pad_dim(mol_type, 0, pad_len)
            res_type = pad_dim(res_type, 0, pad_len)
            disto_center = pad_dim(disto_center, 0, pad_len)
            pad_mask = pad_dim(pad_mask, 0, pad_len)
            resolved_mask = pad_dim(resolved_mask, 0, pad_len)
            disto_mask = pad_dim(disto_mask, 0, pad_len)
            pocket_feature = pad_dim(pocket_feature, 0, pad_len)

    token_features = {
        "token_index": token_index,
        "residue_index": residue_index,
        "asym_id": asym_id,
        "entity_id": entity_id,
        "sym_id": sym_id,
        "mol_type": mol_type,
        "res_type": res_type,
        "disto_center": disto_center,
        "token_pad_mask": pad_mask,
        "token_resolved_mask": resolved_mask,
        "token_disto_mask": disto_mask,
        "pocket_feature": pocket_feature,
    }
    return token_features


def process_atom_features(
    data: Tokenized,
    atoms_per_window_queries: int = 32,
    min_dist: float = 2.0,
    max_dist: float = 22.0,
    num_bins: int = 64,
    max_atoms: Optional[int] = None,
    max_tokens: Optional[int] = None,
    rotation_augment_ref_pos: Optional[bool] = False,
    rotation_augment_coords: Optional[bool] = False,
    center_coords: Optional[torch.Tensor] = None,
) -> dict[str, Tensor]:
    """Get the atom features.

    Parameters
    ----------
    data : Tokenized
        The tokenized data.
    max_atoms : int, optional
        The maximum number of atoms.

    Returns
    -------
    dict[str, Tensor]
        The atom features.

    """
    # Filter to tokens' atoms
    atom_data = []
    ref_space_uid = []
    coord_data = []
    frame_data = []
    resolved_frame_data = []
    atom_to_token = []
    token_to_rep_atom = []  # index on cropped atom table
    r_set_to_rep_atom = []
    disto_coords = []
    atom_idx = 0

    chain_res_ids = {}
    for token_id, token in enumerate(data.tokens):
        # Get the chain residue ids
        chain_idx, res_id = token["asym_id"], token["res_idx"]
        chain = data.structure.chains[chain_idx]

        if (chain_idx, res_id) not in chain_res_ids:
            new_idx = len(chain_res_ids)
            chain_res_ids[(chain_idx, res_id)] = new_idx
        else:
            new_idx = chain_res_ids[(chain_idx, res_id)]

        # Map atoms to token indices
        ref_space_uid.extend([new_idx] * token["atom_num"])
        atom_to_token.extend([token_id] * token["atom_num"])

        # Add atom data
        start = token["atom_idx"]
        end = token["atom_idx"] + token["atom_num"]
        token_atoms = data.structure.atoms[start:end]

        # Map token to representative atom
        token_to_rep_atom.append(atom_idx + token["disto_idx"] - start)
        if (chain["mol_type"] != const.chain_type_ids["NONPOLYMER"]) and token[
            "resolved_mask"
        ]:
            r_set_to_rep_atom.append(atom_idx + token["center_idx"] - start)

        # Get token coordinates
        token_coords = np.array([token_atoms["coords"]])
        coord_data.append(token_coords)

        # Get frame data
        res_type = const.tokens[token["res_type"]]

        if token["atom_num"] < 3 or res_type in ["PAD", "UNK", "-"]:
            idx_frame_a, idx_frame_b, idx_frame_c = 0, 0, 0
            mask_frame = False
        elif (token["mol_type"] == const.chain_type_ids["PROTEIN"]) and (
            res_type in const.ref_atoms
        ):
            idx_frame_a, idx_frame_b, idx_frame_c = (
                const.ref_atoms[res_type].index("N"),
                const.ref_atoms[res_type].index("CA"),
                const.ref_atoms[res_type].index("C"),
            )
            mask_frame = (
                token_atoms["is_present"][idx_frame_a]
                and token_atoms["is_present"][idx_frame_b]
                and token_atoms["is_present"][idx_frame_c]
            )
        elif (
            token["mol_type"] == const.chain_type_ids["DNA"]
            or token["mol_type"] == const.chain_type_ids["RNA"]
        ) and (res_type in const.ref_atoms):
            idx_frame_a, idx_frame_b, idx_frame_c = (
                const.ref_atoms[res_type].index("C1'"),
                const.ref_atoms[res_type].index("C3'"),
                const.ref_atoms[res_type].index("C4'"),
            )
            mask_frame = (
                token_atoms["is_present"][idx_frame_a]
                and token_atoms["is_present"][idx_frame_b]
                and token_atoms["is_present"][idx_frame_c]
            )
        else:
            idx_frame_a, idx_frame_b, idx_frame_c = 0, 0, 0
            mask_frame = False
        frame_data.append(
            [idx_frame_a + atom_idx, idx_frame_b + atom_idx, idx_frame_c + atom_idx]
        )
        resolved_frame_data.append(mask_frame)

        # Get distogram coordinates
        disto_coords_tok = data.structure.atoms[token["disto_idx"]]["coords"]
        disto_coords.append(disto_coords_tok)

        # Update atom data. This is technically never used again (we rely on coord_data),
        # but we update for consistency and to make sure the Atom object has valid, transformed coordinates.
        token_atoms = token_atoms.copy()
        token_atoms["coords"] = token_coords[0]  # atom has a copy of first coords
        atom_data.append(token_atoms)
        atom_idx += len(token_atoms)

    disto_coords = np.array(disto_coords)

    # Compute distogram
    t_center = torch.Tensor(disto_coords)
    t_dists = torch.cdist(t_center, t_center)
    boundaries = torch.linspace(min_dist, max_dist, num_bins - 1)
    distogram = (t_dists.unsqueeze(-1) > boundaries).sum(dim=-1).long()
    disto_target = one_hot(distogram, num_classes=num_bins)

    atom_data = np.concatenate(atom_data)
    coord_data = np.concatenate(coord_data, axis=1)
    ref_space_uid = np.array(ref_space_uid)

    # Compute features
    is_backbone = torch.zeros(len(atom_data), dtype=torch.bool)
    for i, name in enumerate(atom_data["name"]):
        real_name = [chr(c + 32) for c in name if c != 0]
        real_name = "".join(real_name)
        if real_name in ('CA', 'N', 'C', 'O'):
            is_backbone[i] = True
    ref_atom_name_chars = from_numpy(atom_data["name"]).long()
    ref_element = from_numpy(atom_data["element"]).long()
    ref_charge = from_numpy(atom_data["charge"])
    ref_pos = from_numpy(
        atom_data["conformer"].copy()
    )  # not sure why I need to copy here..
    ref_space_uid = from_numpy(ref_space_uid)
    coords = from_numpy(coord_data.copy())
    resolved_mask = from_numpy(atom_data["is_present"])
    pad_mask = torch.ones(len(atom_data), dtype=torch.float)
    atom_to_token = torch.tensor(atom_to_token, dtype=torch.long)
    token_to_rep_atom = torch.tensor(token_to_rep_atom, dtype=torch.long)
    r_set_to_rep_atom = torch.tensor(r_set_to_rep_atom, dtype=torch.long)

    # Convert to one-hot
    ref_atom_name_chars = one_hot(
        ref_atom_name_chars % num_bins, num_classes=num_bins
    )  # added for lower case letters
    ref_element = one_hot(ref_element, num_classes=const.num_elements)
    atom_to_token = one_hot(atom_to_token, num_classes=token_id + 1)
    token_to_rep_atom = one_hot(token_to_rep_atom, num_classes=len(atom_data))
    r_set_to_rep_atom = one_hot(r_set_to_rep_atom, num_classes=len(atom_data))

    # Center the ground truth coordinates
    if center_coords is None:
        center = (coords * resolved_mask[None, :, None]).sum(dim=1)
        center = center / resolved_mask.sum().clamp(min=1)
        coords = coords - center[:, None]
    else:
        coords = coords - center_coords[None, None, :]

    if rotation_augment_ref_pos:
        # Apply random roto-translation to the input atoms
        ref_pos = center_random_augmentation(
            ref_pos[None], resolved_mask[None], centering=False
        )[0]

    if rotation_augment_coords:
        # Apply random roto-translation to the input atoms
        coords = center_random_augmentation(
            coords, resolved_mask[None], centering=False
        )[0]
        coords = coords.unsqueeze(0)

    # Compute padding and apply
    if max_atoms is not None:
        assert max_atoms % atoms_per_window_queries == 0
        pad_len = max_atoms - len(atom_data)
    else:
        pad_len = (
            (len(atom_data) - 1) // atoms_per_window_queries + 1
        ) * atoms_per_window_queries - len(atom_data)

    if pad_len > 0:
        pad_mask = pad_dim(pad_mask, 0, pad_len)
        ref_pos = pad_dim(ref_pos, 0, pad_len)
        resolved_mask = pad_dim(resolved_mask, 0, pad_len)
        ref_element = pad_dim(ref_element, 0, pad_len)
        ref_charge = pad_dim(ref_charge, 0, pad_len)
        ref_atom_name_chars = pad_dim(ref_atom_name_chars, 0, pad_len)
        ref_space_uid = pad_dim(ref_space_uid, 0, pad_len)
        coords = pad_dim(coords, 1, pad_len)
        atom_to_token = pad_dim(atom_to_token, 0, pad_len)
        token_to_rep_atom = pad_dim(token_to_rep_atom, 1, pad_len)
        r_set_to_rep_atom = pad_dim(r_set_to_rep_atom, 1, pad_len)
        is_backbone = pad_dim(is_backbone, 0, pad_len)

    if max_tokens is not None:
        pad_len = max_tokens - token_to_rep_atom.shape[0]
        if pad_len > 0:
            atom_to_token = pad_dim(atom_to_token, 1, pad_len)
            token_to_rep_atom = pad_dim(token_to_rep_atom, 0, pad_len)
            r_set_to_rep_atom = pad_dim(r_set_to_rep_atom, 0, pad_len)
            disto_target = pad_dim(pad_dim(disto_target, 0, pad_len), 1, pad_len)

    return {
        "ref_pos": ref_pos,
        "atom_resolved_mask": resolved_mask,
        "ref_element": ref_element,
        "ref_charge": ref_charge,
        "ref_atom_name_chars": ref_atom_name_chars,
        "ref_space_uid": ref_space_uid,
        "coords": coords,
        "atom_pad_mask": pad_mask,
        "atom_to_token": atom_to_token,
        "is_backbone": is_backbone,
        "token_to_rep_atom": token_to_rep_atom,
        "r_set_to_rep_atom": r_set_to_rep_atom,
        "disto_target": disto_target,
    }


def process_symmetry_features(
    cropped: Tokenized, symmetries: dict
) -> dict[str, Tensor]:
    """Get the symmetry features.

    Parameters
    ----------
    data : Tokenized
        The tokenized data.

    Returns
    -------
    dict[str, Tensor]
        The symmetry features.

    """
    features = get_chain_symmetries(cropped)
    features.update(get_amino_acids_symmetries(cropped))
    features.update(get_ligand_symmetries(cropped, symmetries))

    return features


class BoltzFeaturizer:
    """Boltz featurizer."""

    def process(
        self,
        data: Tokenized,
        atoms_per_window_queries: int = 32,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
        num_bins: int = 64,
        max_tokens: Optional[int] = None,
        max_atoms: Optional[int] = None,
        compute_symmetries: bool = False,
        symmetries: Optional[dict] = None,
        rotation_augment_ref_pos: Optional[bool] = False,
        rotation_augment_coords: Optional[bool] = False,
        center_coords: Optional[torch.Tensor] = None,
    ) -> dict[str, Tensor]:
        """Compute features.

        Parameters
        ----------
        data : Tokenized
            The tokenized data.
        max_tokens : int, optional
            The maximum number of tokens.
        max_atoms : int, optional
            The maximum number of atoms

        Returns
        -------
        dict[str, Tensor]
            The features for model training.

        """

        # Compute token features
        token_features = process_token_features(data, max_tokens)

        # Compute atom features
        atom_features = process_atom_features(
            data,
            atoms_per_window_queries,
            min_dist,
            max_dist,
            num_bins,
            max_atoms,
            max_tokens,
            rotation_augment_ref_pos=rotation_augment_ref_pos,
            rotation_augment_coords=rotation_augment_coords,
            center_coords=center_coords,
        )

        # Compute symmetry features
        symmetry_features = {}
        if compute_symmetries:
            symmetry_features = process_symmetry_features(data, symmetries)

        return {
            **token_features,
            **atom_features,
            **symmetry_features,
        }



def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.reshape(*t.shape, *((1,) * padding_dims))


class BasePath:
    """base class for flow matching path"""

    def __init__(self):
        return

    def compute_alpha_t(self, t):
        """Compute the data coefficient along the path"""
        return None, None

    def compute_sigma_t(self, t):
        """Compute the noise coefficient along the path"""
        return None, None

    def compute_d_alpha_alpha_ratio_t(self, t):
        """Compute the ratio between d_alpha and alpha"""
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        return d_alpha_t / alpha_t

    def compute_mu_t(self, t, x0, x1):
        """Compute the mean of time-dependent density p_t"""
        alpha_t, _ = self.compute_alpha_t(t)
        sigma_t, _ = self.compute_sigma_t(t)
        return alpha_t * x1 + sigma_t * x0

    def compute_xt(self, t, x0, x1):
        """Sample xt from time-dependent density p_t; rng is required"""
        xt = self.compute_mu_t(t, x0, x1)
        return xt

    def compute_ut(self, t, x0, x1):
        """Compute the vector field corresponding to p_t"""
        _, d_alpha_t = self.compute_alpha_t(t)
        _, d_sigma_t = self.compute_sigma_t(t)
        return d_alpha_t * x1 + d_sigma_t * x0

    def interpolant(self, t, x0, x1):
        t = right_pad_dims_to(x0, t)
        xt = self.compute_xt(t, x0, x1)
        ut = self.compute_ut(t, x0, x1)
        return t, xt, ut

    def compute_drift(self, x, t):
        """We always output sde according to score parametrization; """
        t = right_pad_dims_to(x, t)
        alpha_ratio = self.compute_d_alpha_alpha_ratio_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        drift_mean = alpha_ratio * x
        drift_var = alpha_ratio * (sigma_t ** 2) - sigma_t * d_sigma_t
        return -drift_mean, drift_var

    def compute_score_from_velocity(self, v_t, y_t, t):
        t = right_pad_dims_to(y_t, t)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        mean = y_t
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
        score = (reverse_alpha_ratio * v_t - mean) / var
        return score

    def compute_velocity_from_score(self, s_t, y_t, t):
        t = right_pad_dims_to(y_t, t)
        drift_mean, drift_var = self.compute_drift(y_t, t)
        velocity = -drift_mean + drift_var * s_t
        return velocity


class LinearPath(BasePath):
    """
    Linear flow process:
    x0: noise, x1: data
    In inference, we sample data from 0 -> 1
    """

    def __init__(self):
        super().__init__()

    def compute_alpha_t(self, t):
        """Compute the data coefficient along the path"""
        return t, 1

    def compute_sigma_t(self, t):
        """Compute the noise coefficient along the path"""
        return 1 - t, -1

    def compute_d_alpha_alpha_ratio_t(self, t):
        """Compute the ratio between d_alpha and alpha"""
        return 1 / t


class ProteinDataProcessor:
    def __init__(
        self, 
        device, 
        scale=16.0, 
        ref_scale=5.0, 
        multiplicity=1,
        inference_multiplicity=1,
        backend="torch",
    ):
        self.device = device
        self.scale = scale
        self.ref_scale = ref_scale
        # if multiplicity > 1, effective batch size is multiplicity * batch_size
        self.multiplicity = multiplicity
        self.inference_multiplicity = inference_multiplicity
        self.backend = backend

        self.center_random_fn = center_random_augmentation

    def process_esm(
        self, 
        batch, 
        esm_model=None, 
        esm_dict=None, 
        af2_to_esm=None,
        inference=False,
    ):
        sequence = batch["aa_seq"]
        B = len(sequence)
        L = batch["res_type"].shape[1]
        num_tokens = batch["cropped_num_tokens"]

        aatype, mask, residx, linker_mask, _ = batch_encode_sequences(
            sequence, residue_index_offset=512, chain_linker="G" * 25,
        )

        aatype, mask, residx, linker_mask = map(
            lambda x: x.to(self.device), (aatype, mask, residx, linker_mask)
        )

        if residx is None:
            residx = torch.arange(L, device=self.device).expand_as(aatype)

        esmaa = af2_idx_to_esm_idx(aatype, mask, af2_to_esm)

        multiplicity = self.multiplicity if not inference else self.inference_multiplicity

        esm_s_, _ = compute_language_model_representations(
            esmaa, esm_model, esm_dict, backend=self.backend
        )

        esm_s_ = esm_s_.detach()
        mask, linker_mask = mask.detach().bool(), linker_mask.detach().bool()

        if multiplicity > 1:
            true_mask = linker_mask & mask
            true_len = true_mask[0].sum()
            assert true_len == num_tokens[0]
            esm_s = torch.zeros(
                (1, L, esm_model.num_layers + 1, esm_s_.shape[-1]),
                device=self.device,
            )
            esm_s[0, :true_len] = esm_s_[0, true_mask[0]]
            esm_s = esm_s.repeat_interleave(multiplicity, dim=0)
        else:
            esm_s = torch.zeros(
                (B, L, esm_model.num_layers + 1, esm_s_.shape[-1]),
                device=self.device,
            )
            true_mask = linker_mask & mask
            for i in range(B):
                true_len = true_mask[i].sum()
                assert true_len == num_tokens[i]
                esm_s[i, :true_len] = esm_s_[i, true_mask[i]]

        batch["esm_s"] = esm_s

        return

    def batch_to_device(self, batch, multiplicity=1):
        for k, v in batch.items():
            # if isinstance(v, torch.Tensor) and k in key2cuda:
            if isinstance(v, torch.Tensor):
                if multiplicity > 1:
                    v = v.repeat_interleave(multiplicity, dim=0)
                batch[k] = v.to(self.device)
        return batch


    def preprocess_training(self, batch, esm_model=None, esm_dict=None, af2_to_esm=None):
        batch_size, max_ntokens = batch["mol_type"].shape[:2]
        max_natoms = batch["ref_element"].shape[1]

        batch['atom_to_token_idx'] = torch.argmax(
            batch['atom_to_token'], dim=-1)

        y = batch['coords'].float().squeeze(1) / self.scale
        batch['coords'] = y

        ref_y = batch['ref_pos'].float() / self.ref_scale
        batch['ref_pos'] = ref_y

        mol_index = torch.arange(max_natoms).unsqueeze(0).expand(
            batch_size, -1)
        batch['mol_index'] = mol_index

        batch = self.batch_to_device(batch, multiplicity=self.multiplicity)

        if esm_model is not None:
            self.process_esm(batch, esm_model, esm_dict, af2_to_esm)

        # randomly augment the coordinates if repeating batch
        if self.multiplicity > 1:
            batch['coords'] = self.center_random_fn(
                batch['coords'], 
                batch['atom_pad_mask'], 
                centering=True,
                augmentation=True,
            )

        return batch

    def preprocess_inference(self, batch, esm_model=None, esm_dict=None, af2_to_esm=None):
        batch_size, max_ntokens = batch["mol_type"].shape[:2]
        max_natoms = batch["ref_element"].shape[1]

        batch['coords'] = batch['coords'].squeeze(1) / self.scale
        batch['ref_pos'] = batch['ref_pos'].float() / self.ref_scale

        batch['atom_to_token_idx'] = torch.argmax(
            batch['atom_to_token'], dim=-1)

        mol_index = torch.arange(max_natoms).unsqueeze(0).expand(
            batch_size, -1)
        batch['mol_index'] = mol_index

        batch = self.batch_to_device(batch, multiplicity=self.inference_multiplicity)

        if esm_model is not None and batch.get('esm_s', None) is None:
            print("Processing ESM features for inference...")
            self.process_esm(batch, esm_model, esm_dict, af2_to_esm, inference=True)

        # MLX branch disabled

        return batch

    def postprocess(self, out_dict, batch):
        out_dict['coords'] = self.center_random_fn(
            batch['coords'], 
            batch['atom_pad_mask'], 
            centering=True,
            augmentation=False,
        ) * self.scale
        out_dict['denoised_coords'] = self.center_random_fn(
            out_dict['denoised_coords'], 
            batch['atom_pad_mask'], 
            centering=True,
            augmentation=False,
        ) * self.scale
        return out_dict


def center_random_augmentation(
    atom_coords,
    atom_mask,
    s_trans=1.0,
    augmentation=True,
    centering=True,
    return_second_coords=False,
    second_coords=None,
):
    """Center and randomly augment the input coordinates.

    Parameters
    ----------
    atom_coords : Tensor
        The atom coordinates.
    atom_mask : Tensor
        The atom mask.
    s_trans : float, optional
        The translation factor, by default 1.0
    augmentation : bool, optional
        Whether to add rotational and translational augmentation the input, by default True
    centering : bool, optional
        Whether to center the input, by default True

    Returns
    -------
    Tensor
        The augmented atom coordinates.

    """
    if centering:
        atom_mean = torch.sum(
            atom_coords * atom_mask[:, :, None], dim=1, keepdim=True
        ) / torch.sum(atom_mask[:, :, None], dim=1, keepdim=True)
        atom_coords = atom_coords - atom_mean

        if second_coords is not None:
            # apply same transformation also to this input
            second_coords = second_coords - atom_mean

    if augmentation:
        # simple no-op rotation placeholder for torch-only path
        # (kept to preserve interface; no rotation applied)
        random_trans = torch.randn_like(atom_coords[:, 0:1, :]) * s_trans
        atom_coords = atom_coords + random_trans

        if second_coords is not None:
            second_coords = second_coords + random_trans

    if return_second_coords:
        return atom_coords, second_coords

    return atom_coords


def logit_normal_sample(n=1, m=0.0, s=1.0):
    # Logit-Normal Sampling from https://arxiv.org/pdf/2403.03206.pdf
    u = torch.randn(n) * s + m
    t = 1 / (1 + torch.exp(-u))
    return t


def lddt_dist(dmat_predicted, dmat_true, mask, cutoff=15.0, per_atom=False):
    # NOTE: the mask is a pairwise mask which should have the identity elements already masked out
    # Compute mask over distances
    dists_to_score = (dmat_true < cutoff).float() * mask
    dist_l1 = torch.abs(dmat_true - dmat_predicted)

    score = 0.25 * (
        (dist_l1 < 0.5).float()
        + (dist_l1 < 1.0).float()
        + (dist_l1 < 2.0).float()
        + (dist_l1 < 4.0).float()
    )

    # Normalize over the appropriate axes.
    if per_atom:
        mask_no_match = torch.sum(dists_to_score, dim=-1) != 0
        norm = 1.0 / (1e-10 + torch.sum(dists_to_score, dim=-1))
        score = norm * (1e-10 + torch.sum(dists_to_score * score, dim=-1))
        return score, mask_no_match.float()
    else:
        norm = 1.0 / (1e-10 + torch.sum(dists_to_score, dim=(-2, -1)))
        score = norm * (1e-10 + torch.sum(dists_to_score * score, dim=(-2, -1)))
        total = torch.sum(dists_to_score, dim=(-1, -2))
        return score, total


def process_structure(
    data: PDB,
    resource: Resource,
    out_dir: Path,
    filters: list[StaticFilter],
    clusters: dict,
) -> None:
    """Process a target.

    Parameters
    ----------
    item : PDB
        The raw input data.
    resource: Resource
        The shared resource.
    out_dir : Path
        The output directory.

    """
    # Check if we need to process
    struct_path = out_dir / "structures" / f"{data.id}.npz"
    record_path = out_dir / "records" / f"{data.id}.json"

    if struct_path.exists() and record_path.exists():
        return

    try:
        # Parse the target
        target: Target = parse(data, resource, clusters)
        structure = target.structure

        # Apply the filters
        mask = structure.mask
        if filters is not None:
            for f in filters:
                filter_mask = f.filter(structure)
                mask = mask & filter_mask
    except Exception:
        traceback.print_exc()
        print(f"Failed to parse {data.id}")
        return

    # Replace chains and interfaces
    chains = []
    for i, chain in enumerate(target.record.chains):
        chains.append(replace(chain, valid=bool(mask[i])))

    # Replace structure and record
    structure = replace(structure, mask=mask)
    record = replace(target.record, chains=chains, interfaces=[])
    target = replace(target, structure=structure, record=record)

    # Dump structure
    np.savez_compressed(struct_path, **asdict(structure))

    # Dump record
    with record_path.open("w") as f:
        json.dump(asdict(record), f)


class EMSampler():
    """
    A Euler-Maruyama solver for SDEs.
    """
    def __init__(
        self,
        num_timesteps=500,
        t_start=1e-4,
        tau=0.3,
        log_timesteps=False,
        w_cutoff=0.99,
    ):
        self.num_timesteps = num_timesteps
        self.log_timesteps = log_timesteps
        self.t_start = t_start
        self.tau = tau
        self.w_cutoff = w_cutoff

        if self.log_timesteps:
            t = 1.0 - torch.logspace(-2, 0, self.num_timesteps + 1).flip(0)
            t = t - torch.min(t)
            t = t / torch.max(t)
            self.steps = t.clamp(min=self.t_start, max=1.0)
        else:
            self.steps = torch.linspace(
                self.t_start, 1.0, steps=self.num_timesteps + 1
            )

    def diffusion_coefficient(self, t, eps=0.01):
        # determine diffusion coefficient
        w = (1.0 - t) / (t + eps)
        if t >= self.w_cutoff:
            w = 0.0
        return w

    @torch.no_grad()
    def euler_maruyama_step(
        self,
        model_fn,
        flow,
        y, 
        t, 
        t_next, 
        batch, 
    ):
        dt = t_next - t
        eps = torch.randn_like(y).to(y)

        y = center_random_augmentation(
            y,
            batch["atom_pad_mask"],
            augmentation=False,
            centering=True,
        )

        batched_t = repeat(t, " -> b", b=y.shape[0])
        velocity = model_fn(
            noised_pos=y,
            t=batched_t,
            feats=batch,
        )['predict_velocity']
        score = flow.compute_score_from_velocity(velocity, y, t)

        diff_coeff = self.diffusion_coefficient(t)
        drift = velocity + diff_coeff * score
        mean_y = y + drift * dt
        y_sample = mean_y + torch.sqrt(2.0 * dt * diff_coeff * self.tau) * eps

        return y_sample

    @torch.no_grad()
    def sample(self, model_fn, flow, noise, batch):
        sampling_timesteps = self.num_timesteps
        steps = self.steps.to(noise.device)
        y_sampled = noise
        feats = batch

        for i in tqdm(
            range(sampling_timesteps),
            desc="Sampling",
            total=sampling_timesteps,
        ):
            t = steps[i]
            t_next = steps[i + 1]

            y_sampled = self.euler_maruyama_step(
                model_fn,
                flow,
                y_sampled,
                t,
                t_next,
                feats,
            )

        return {
            "denoised_coords": y_sampled
        }


class ConfidenceModule(nn.Module):
    def __init__(
        self,
        hidden_size,
        transformer_blocks,
        num_plddt_bins=50,
    ):
        super().__init__()
        self.transformer_blocks = transformer_blocks
        self.to_plddt_logits = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, num_plddt_bins),
        )

    def forward(
        self,
        latent,
        feats,
    ):
        token_pe_pos = torch.cat(
            [
                feats["residue_index"].unsqueeze(-1).float(),        # (B, M, 1)
                feats["entity_id"].unsqueeze(-1).float(),            # (B, M, 1)
                feats["asym_id"].unsqueeze(-1).float(),              # (B, M, 1)
                feats["sym_id"].unsqueeze(-1).float(),               # (B, M, 1)
            ],
            dim=-1,
        )  

        latent = self.transformer_blocks(
            latents=latent, 
            c=None,
            pos=token_pe_pos,
        )

        # Compute the pLDDT
        plddt_logits = self.to_plddt_logits(latent)

        # Compute the aggregated pLDDT
        plddt = compute_aggregated_metric(plddt_logits)

        return dict(
            plddt=plddt,
            plddt_logits=plddt_logits,
        )


class SelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_bias=True,
        qk_norm=True,
        pos_embedder=None,
        linear_target: nn.Module = nn.Linear,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = linear_target(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = linear_target(hidden_size, hidden_size, bias=use_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.pos_embedder = pos_embedder

    def forward(self, x, **kwargs):
        B, N, C = x.shape
        attn_mask = kwargs.get("attention_mask")
        pos = kwargs.get("pos")

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = rearrange(qkv, "b n t h c -> t b h n c")
        q, k, v = qkv.unbind(0)

        if attn_mask is not None:
            attn_mask = attn_mask.to(dtype=q.dtype)

        if self.pos_embedder and pos is not None:
            q, k = self.pos_embedder(q, k, pos)

        q, k = self.q_norm(q), self.k_norm(k)
        x = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of=256):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=True)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w1.weight)
        torch.nn.init.xavier_uniform_(self.w2.weight)
        torch.nn.init.xavier_uniform_(self.w3.weight)
        if self.w1.bias is not None:
            torch.nn.init.constant_(self.w1.bias, 0)
        if self.w2.bias is not None:
            torch.nn.init.constant_(self.w2.bias, 0)
        if self.w3.bias is not None:
            torch.nn.init.constant_(self.w3.bias, 0)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ConditionEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, input_dim, hidden_size, dropout_prob):
        super().__init__()
        self.proj = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.SiLU(),
            )
        self.dropout_prob = dropout_prob
        self.null_token = nn.Parameter(torch.randn(input_dim), requires_grad=True)

    def token_drop(self, cond, force_drop_ids=None):
        """
        cond: (B, N, D)
        Drops conditions to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(cond.shape[0], device=cond.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids
        cond[drop_ids] = self.null_token[None, None, :]
        return cond

    def forward(self, cond, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            cond = self.token_drop(cond, force_drop_ids)
        embeddings = self.proj(cond)
        return embeddings


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, out_channels, c_dim=None):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(c_dim, 2 * hidden_size, bias=True)
        )
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Zero-out output layers:
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class RMSNorm(nn.Module):
    def __init__(self, d, p=-1.0, eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0.0 or self.p > 1.0:
            norm_x = x.norm(2, dim=-1, keepdim=True, dtype=x.dtype)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True, dtype=x.dtype)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1.0 / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, cfg: "SimpleFoldConfig", hidden_size: int, pos_embedder: T.Optional[nn.Module] = None, num_heads_override: T.Optional[int] = None):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(
            hidden_size=hidden_size,
            num_heads=(cfg.num_heads if num_heads_override is None else num_heads_override),
            qkv_bias=cfg.qkv_bias,
            qk_scale=None,
            attn_drop=cfg.attn_drop,
            proj_drop=cfg.proj_drop,
            use_bias=cfg.attn_use_bias,
            qk_norm=cfg.qk_norm,
            pos_embedder=pos_embedder,
            linear_target=nn.Linear,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * cfg.mlp_ratio)
        if cfg.use_swiglu:
            self.mlp = SwiGLU(hidden_size, mlp_hidden_dim)
        else:
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0,
            )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Zero-out adaLN modulation layers in DiT encoder blocks:
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(
        self,
        latents,
        c,
        **kwargs,
    ):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        _latents = self.attn(
            modulate(self.norm1(latents), shift_msa, scale_msa), **kwargs
        )
        latents = latents + gate_msa.unsqueeze(1) * _latents
        latents = latents + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(latents), shift_mlp, scale_mlp)
        )
        return latents


class TransformerBlock(nn.Module):
    """
    A standard Transformer block (no adaLN conditioning).
    """

    def __init__(self, cfg: "SimpleFoldConfig", hidden_size: int, pos_embedder: T.Optional[nn.Module] = None, num_heads_override: T.Optional[int] = None):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(
            hidden_size=hidden_size,
            num_heads=(cfg.num_heads if num_heads_override is None else num_heads_override),
            qkv_bias=cfg.qkv_bias,
            qk_scale=None,
            attn_drop=cfg.attn_drop,
            proj_drop=cfg.proj_drop,
            use_bias=cfg.attn_use_bias,
            qk_norm=cfg.qk_norm,
            pos_embedder=pos_embedder,
            linear_target=nn.Linear,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * cfg.mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        if cfg.use_swiglu:
            self.mlp = SwiGLU(hidden_size, mlp_hidden_dim)
        else:
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0,
            )

    def forward(
        self,
        latents,
        **kwargs,
    ):
        _latents = self.attn(self.norm1(latents), **kwargs)
        latents = latents + _latents
        latents = latents + self.mlp(self.norm2(latents))
        return latents


class HomogenTrunk(nn.Module):
    def __init__(self, block_ctor: T.Callable[[], nn.Module], depth: int):
        super().__init__()
        self.blocks = nn.ModuleList([block_ctor() for _ in range(depth)])

    def forward(self, latents, c, **kwargs):
        for i, block in enumerate(self.blocks):
            kwargs["layer_idx"] = i
            latents = block(latents=latents, c=c, **kwargs)
        return latents


class StackedDiT(nn.Module):
    def __init__(
        self,
        cfg: "SimpleFoldConfig",
        hidden_size: int,
        depth: int,
        num_heads_override: T.Optional[int] = None,
        pos_embedder: T.Optional[nn.Module] = None,
    ):
        super().__init__()
        def ctor():
            return DiTBlock(cfg=cfg, hidden_size=hidden_size, pos_embedder=pos_embedder, num_heads_override=num_heads_override)
        self.trunk = HomogenTrunk(block_ctor=ctor, depth=depth)

    def forward(self, latents, c, **kwargs):
        return self.trunk(latents=latents, c=c, **kwargs)


class SimpleFoldUtils:
    def __init__(self):
        pass

    def create_local_attn_bias(
        self, n: int, n_queries: int, n_keys: int, inf: float = 1e10, device: torch.device = None
    ) -> torch.Tensor:
        """Create local attention bias based on query window n_queries and kv window n_keys.

        Args:
            n (int): the length of quiries
            n_queries (int): window size of quiries
            n_keys (int): window size of keys/values
            inf (float, optional): the inf to mask attention. Defaults to 1e10.
            device (torch.device, optional): cuda|cpu|None. Defaults to None.

        Returns:
            torch.Tensor: the diagonal-like global attention bias
        """
        n_trunks = int(math.ceil(n / n_queries))
        padded_n = n_trunks * n_queries
        attn_mask = torch.zeros(padded_n, padded_n, device=device)
        for block_index in range(0, n_trunks):
            i = block_index * n_queries
            j1 = max(0, n_queries * block_index - (n_keys - n_queries) // 2)
            j2 = n_queries * block_index + (n_queries + n_keys) // 2
            attn_mask[i : i + n_queries, j1:j2] = 1.0
        attn_bias = (1 - attn_mask) * -inf
        return attn_bias.to(device=device)[:n, :n]

    def create_atom_attn_mask(
        self, 
        feats, 
        natoms, 
        atom_n_queries=None, 
        atom_n_keys=None,
        inf: float = 1e10
    ) -> torch.Tensor:
        if atom_n_queries is not None and atom_n_keys is not None:
            atom_attn_mask = self.create_local_attn_bias(
                n=natoms,
                n_queries=atom_n_queries,
                n_keys=atom_n_keys,
                device=feats["ref_pos"].device,
                inf=inf,
            )
        else:
            atom_attn_mask = None

        return atom_attn_mask

    def loss_masking(self, loss, atom_mask):
        loss_mask = repeat(atom_mask, "b s -> b s d", d=loss.shape[-1])
        loss *= loss_mask

        denom = torch.sum(atom_mask, -1, keepdim=True)
        denom = denom.unsqueeze(-1)
        loss = torch.sum(loss, dim=1, keepdim=True) / denom
        return loss

    def smooth_lddt_loss(
        self, 
        pred_coords,
        true_coords,
        # is_nucleotide,
        coords_mask,
        t,
    ):
        """Compute weighted alignment.

        Parameters
        ----------
        pred_coords: torch.Tensor
            The predicted atom coordinates
        true_coords: torch.Tensor
            The ground truth atom coordinates
        coords_mask: torch.Tensor
            The atoms mask

        """
        B, N, _ = true_coords.shape
        true_dists = torch.cdist(true_coords, true_coords)

        mask = (true_dists < self.lddt_cutoff).float()
        mask = mask * (1 - torch.eye(pred_coords.shape[1], device=pred_coords.device))
        mask = mask * (coords_mask.unsqueeze(-1) * coords_mask.unsqueeze(-2))

        # Compute distances between all pairs of atoms
        pred_dists = torch.cdist(pred_coords, pred_coords)
        dist_diff = torch.abs(true_dists - pred_dists)

        # Compute epsilon values
        eps = (
            (
                (
                    F.sigmoid(0.5 - dist_diff)
                    + F.sigmoid(1.0 - dist_diff)
                    + F.sigmoid(2.0 - dist_diff)
                    + F.sigmoid(4.0 - dist_diff)
                )
                / 4.0
            )
            .view(B, N, N)
            .mean(dim=0)
        )

        # Calculate masked averaging
        num = (eps * mask).sum(dim=(-1, -2))
        den = mask.sum(dim=(-1, -2)).clamp(min=1)
        lddt = num / den
        if self.lddt_weight_schedule:
            t_weight = 1 + 8 * torch.relu(t - 0.5)
            lddt = (1.0 - lddt) * t_weight
            return lddt.mean()
        else:
            return (1.0 - lddt.mean()) * self.smooth_lddt_loss_weight

    def plddt_loss(
        self,
        pred_lddt,
        pred_atom_coords,
        true_atom_coords,
        true_coords_resolved_mask,
        feats,
        # multiplicity=1,
    ):
        """Compute plddt loss.

        Parameters
        ----------
        pred_lddt: torch.Tensor
            The plddt logits
        pred_atom_coords: torch.Tensor
            The predicted atom coordinates
        true_atom_coords: torch.Tensor
            The atom coordinates after symmetry correction
        true_coords_resolved_mask: torch.Tensor
            The resolved mask after symmetry correction
        feats: Dict[str, torch.Tensor]
            Dictionary containing the model input

        Returns
        -------
        torch.Tensor
            Plddt loss

        """

        # extract necessary features
        atom_mask = true_coords_resolved_mask

        R_set_to_rep_atom = feats["r_set_to_rep_atom"].float()
        # R_set_to_rep_atom = R_set_to_rep_atom.repeat_interleave(multiplicity, 0).float()

        token_type = feats["mol_type"]
        # token_type = token_type.repeat_interleave(multiplicity, 0)
        # is_nucleotide_token = (token_type == const.chain_type_ids["DNA"]).float() + (
        #     token_type == const.chain_type_ids["RNA"]
        # ).float()

        B = true_atom_coords.shape[0]

        # atom_to_token = feats["atom_to_token"].float()
        # atom_to_token = atom_to_token.repeat_interleave(multiplicity, 0)

        token_to_rep_atom = feats["token_to_rep_atom"].float()
        # token_to_rep_atom = token_to_rep_atom.repeat_interleave(multiplicity, 0)

        true_token_coords = torch.bmm(token_to_rep_atom, true_atom_coords)
        pred_token_coords = torch.bmm(token_to_rep_atom, pred_atom_coords)

        # compute true lddt
        true_d = torch.cdist(
            true_token_coords,
            torch.bmm(R_set_to_rep_atom, true_atom_coords),
        )
        pred_d = torch.cdist(
            pred_token_coords,
            torch.bmm(R_set_to_rep_atom, pred_atom_coords),
        )

        # compute mask
        pair_mask = atom_mask.unsqueeze(-1) * atom_mask.unsqueeze(-2)
        pair_mask = (
            pair_mask
            * (1 - torch.eye(pair_mask.shape[1], device=pair_mask.device))[None, :, :]
        )
        pair_mask = torch.einsum("bnm,bkm->bnk", pair_mask, R_set_to_rep_atom)
        pair_mask = torch.bmm(token_to_rep_atom, pair_mask)
        atom_mask = torch.bmm(token_to_rep_atom, atom_mask.unsqueeze(-1).float())
        # is_nucleotide_R_element = torch.bmm(
        #     R_set_to_rep_atom, torch.bmm(atom_to_token, is_nucleotide_token.unsqueeze(-1))
        # ).squeeze(-1)
        # cutoff = 15 + 15 * is_nucleotide_R_element.reshape(B, 1, -1).repeat(
        #     1, true_d.shape[1], 1
        # )

        # compute lddt
        target_lddt, mask_no_match = lddt_dist(
            pred_d, true_d, pair_mask, cutoff=15.0, per_atom=True
        )

        # compute loss
        num_bins = pred_lddt.shape[-1]
        bin_index = torch.floor(target_lddt * num_bins).long()
        bin_index = torch.clamp(bin_index, max=(num_bins - 1))
        lddt_one_hot = F.one_hot(bin_index, num_classes=num_bins)
        errors = -1 * torch.sum(
            lddt_one_hot * F.log_softmax(pred_lddt, dim=-1),
            dim=-1,
        )
        atom_mask = atom_mask.squeeze(-1)
        loss = torch.sum(errors * atom_mask * mask_no_match, dim=-1) / (
            1e-7 + torch.sum(atom_mask * mask_no_match, dim=-1)
        )

        # Average over the batch dimension
        loss = torch.mean(loss)

        self.log(
            "loss/plddt",
            loss.item(),
            on_epoch=True,
            logger=True,
            prog_bar=True,
            rank_zero_only=True,
        )

        return loss

    def save_result(self, structure, record, results, out_name):
        sampled_coord = results["sampled_coord"]
        pad_mask = results["pad_mask"]
        plddt = results["plddts"]

        save_paths = []
        for i in range(sampled_coord.shape[0]):
            sampled_coord_i = sampled_coord[i]
            pad_mask_i = pad_mask[i]
            plddt_i = plddt[i] if plddt is not None else None
            out_name_i = f"{out_name}_sampled_{i}"
            # save the generated structure
            structure_save = process_structure(
                structure, sampled_coord_i, pad_mask_i, record, backend=self.backend
            )
            save_structure(
                structure_save,
                self.prediction_dir,
                out_name_i,
                output_format="mmcif",
                plddts=plddt_i,
            )
            save_paths.append(self.prediction_dir / f"{out_name_i}.cif")
        return save_paths


def to_pdb(structure: Structure, plddts: Optional[Tensor] = None) -> str:  # noqa: PLR0915
    """Write a structure into a PDB file.

    Parameters
    ----------
    structure : Structure
        The input structure

    Returns
    -------
    str
        the output PDB file

    """
    pdb_lines = []

    atom_index = 1
    atom_reindex_ter = []
    chain_tags = generate_tags()

    # Load periodic table for element mapping
    periodic_table = Chem.GetPeriodicTable()

    # Add all atom sites.
    res_num = 0
    for chain in structure.chains:
        # We rename the chains in alphabetical order
        chain_idx = chain["asym_id"]
        chain_tag = next(chain_tags)

        res_start = chain["res_idx"]
        res_end = chain["res_idx"] + chain["res_num"]

        residues = structure.residues[res_start:res_end]
        for residue in residues:
            atom_start = residue["atom_idx"]
            atom_end = residue["atom_idx"] + residue["atom_num"]
            atoms = structure.atoms[atom_start:atom_end]
            atom_coords = atoms["coords"]
            for i, atom in enumerate(atoms):
                # This should not happen on predictions, but just in case.
                if not atom["is_present"]:
                    continue

                record_type = (
                    "ATOM"
                    if chain["mol_type"] != const.chain_type_ids["NONPOLYMER"]
                    else "HETATM"
                )
                name = atom["name"]
                name = [chr(c + 32) for c in name if c != 0]
                name = "".join(name)
                name = name if len(name) == 4 else f" {name}"  # noqa: PLR2004
                alt_loc = ""
                insertion_code = ""
                occupancy = 1.00
                element = periodic_table.GetElementSymbol(atom["element"].item())
                element = element.upper()
                charge = ""
                residue_index = residue["res_idx"] + 1
                pos = atom_coords[i]
                res_name_3 = (
                    "LIG" if record_type == "HETATM" else str(residue["name"][:3])
                )
                b_factor = (
                    100.00 if plddts is None else round(plddts[res_num].item(), 2)
                )

                # PDB is a columnar format, every space matters here!
                atom_line = (
                    f"{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}"
                    f"{res_name_3:>3} {chain_tag:>1}"
                    f"{residue_index:>4}{insertion_code:>1}   "
                    f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                    f"{occupancy:>6.2f}{b_factor:>6.2f}          "
                    f"{element:>2}{charge:>2}"
                )
                pdb_lines.append(atom_line)
                atom_reindex_ter.append(atom_index)
                atom_index += 1

            res_num += 1

        should_terminate = chain_idx < (len(structure.chains) - 1)
        if should_terminate:
            # Close the chain.
            chain_end = "TER"
            chain_termination_line = (
                f"{chain_end:<6}{atom_index:>5}      "
                f"{res_name_3:>3} "
                f"{chain_tag:>1}{residue_index:>4}"
            )
            pdb_lines.append(chain_termination_line)
            atom_index += 1

    # Dump CONECT records.
    for bonds in [structure.bonds, structure.connections]:
        for bond in bonds:
            atom1 = structure.atoms[bond["atom_1"]]
            atom2 = structure.atoms[bond["atom_2"]]
            if not atom1["is_present"] or not atom2["is_present"]:
                continue
            atom1_idx = atom_reindex_ter[bond["atom_1"]]
            atom2_idx = atom_reindex_ter[bond["atom_2"]]
            conect_line = f"CONECT{atom1_idx:>5}{atom2_idx:>5}"
            pdb_lines.append(conect_line)

    pdb_lines.append("END")
    pdb_lines.append("")
    pdb_lines = [line.ljust(80) for line in pdb_lines]
    return "\n".join(pdb_lines)


def to_mmcif(structure: Structure, plddts: Optional[Tensor] = None) -> str:  # noqa: C901, PLR0915, PLR0912
    """Write a structure into an MMCIF file.

    Parameters
    ----------
    structure : Structure
        The input structure

    Returns
    -------
    str
        the output MMCIF file

    """
    system = System()

    # Load periodic table for element mapping
    periodic_table = Chem.GetPeriodicTable()

    # Map entities to chain_ids
    entity_to_chains = {}
    entity_to_moltype = {}

    for chain in structure.chains:
        entity_id = chain["entity_id"]
        mol_type = chain["mol_type"]
        entity_to_chains.setdefault(entity_id, []).append(chain)
        entity_to_moltype[entity_id] = mol_type

    # Map entities to sequences
    sequences = {}
    for entity in entity_to_chains:
        # Get the first chain
        chain = entity_to_chains[entity][0]

        # Get the sequence
        res_start = chain["res_idx"]
        res_end = chain["res_idx"] + chain["res_num"]
        residues = structure.residues[res_start:res_end]
        sequence = [str(res["name"]) for res in residues]
        sequences[entity] = sequence

    # Create entity objects
    lig_entity = None
    entities_map = {}
    for entity, sequence in sequences.items():
        mol_type = entity_to_moltype[entity]

        if mol_type == const.chain_type_ids["PROTEIN"]:
            alphabet = ihm.LPeptideAlphabet()
            chem_comp = lambda x: ihm.LPeptideChemComp(id=x, code=x, code_canonical="X")  # noqa: E731
        elif mol_type == const.chain_type_ids["DNA"]:
            alphabet = ihm.DNAAlphabet()
            chem_comp = lambda x: ihm.DNAChemComp(id=x, code=x, code_canonical="N")  # noqa: E731
        elif mol_type == const.chain_type_ids["RNA"]:
            alphabet = ihm.RNAAlphabet()
            chem_comp = lambda x: ihm.RNAChemComp(id=x, code=x, code_canonical="N")  # noqa: E731
        elif len(sequence) > 1:
            alphabet = {}
            chem_comp = lambda x: ihm.SaccharideChemComp(id=x)  # noqa: E731
        else:
            alphabet = {}
            chem_comp = lambda x: ihm.NonPolymerChemComp(id=x)  # noqa: E731

        # Handle smiles
        if len(sequence) == 1 and (sequence[0] == "LIG"):
            if lig_entity is None:
                seq = [chem_comp(sequence[0])]
                lig_entity = Entity(seq)
            model_e = lig_entity
        else:
            seq = [
                alphabet[item] if item in alphabet else chem_comp(item)
                for item in sequence
            ]
            model_e = Entity(seq)

        for chain in entity_to_chains[entity]:
            chain_idx = chain["asym_id"]
            entities_map[chain_idx] = model_e

    # We don't assume that symmetry is perfect, so we dump everything
    # into the asymmetric unit, and produce just a single assembly
    chain_tags = generate_tags()
    asym_unit_map = {}
    for chain in structure.chains:
        # Define the model assembly
        chain_idx = chain["asym_id"]
        chain_tag = next(chain_tags)
        asym = AsymUnit(
            entities_map[chain_idx],
            details="Model subunit %s" % chain_tag,
            id=chain_tag,
        )
        asym_unit_map[chain_idx] = asym
    modeled_assembly = Assembly(asym_unit_map.values(), name="Modeled assembly")

    class _LocalPLDDT(modelcif.qa_metric.Local, modelcif.qa_metric.PLDDT):
        name = "pLDDT"
        software = None
        description = "Predicted lddt"

    class _MyModel(AbInitioModel):
        def get_atoms(self) -> Iterator[Atom]:
            # Add all atom sites.
            res_num = 0
            for chain in structure.chains:
                # We rename the chains in alphabetical order
                het = chain["mol_type"] == const.chain_type_ids["NONPOLYMER"]
                chain_idx = chain["asym_id"]
                res_start = chain["res_idx"]
                res_end = chain["res_idx"] + chain["res_num"]

                residues = structure.residues[res_start:res_end]
                for residue in residues:
                    atom_start = residue["atom_idx"]
                    atom_end = residue["atom_idx"] + residue["atom_num"]
                    atoms = structure.atoms[atom_start:atom_end]
                    atom_coords = atoms["coords"]
                    for i, atom in enumerate(atoms):
                        # This should not happen on predictions, but just in case.
                        if not atom["is_present"]:
                            continue

                        name = atom["name"]
                        name = [chr(c + 32) for c in name if c != 0]
                        name = "".join(name)
                        element = periodic_table.GetElementSymbol(
                            atom["element"].item()
                        )
                        element = element.upper()
                        residue_index = residue["res_idx"] + 1
                        pos = atom_coords[i]
                        biso = (
                            100.00
                            if plddts is None
                            else round(plddts[res_num].item(), 2)
                        )
                        yield Atom(
                            asym_unit=asym_unit_map[chain_idx],
                            type_symbol=element,
                            seq_id=residue_index,
                            atom_id=name,
                            x=f"{pos[0]:.5f}",
                            y=f"{pos[1]:.5f}",
                            z=f"{pos[2]:.5f}",
                            het=het,
                            biso=biso,
                            occupancy=1,
                        )

                    res_num += 1

        def add_plddt(self, plddts):
            res_num = 0
            for chain in structure.chains:
                chain_idx = chain["asym_id"]
                res_start = chain["res_idx"]
                res_end = chain["res_idx"] + chain["res_num"]
                residues = structure.residues[res_start:res_end]
                # We rename the chains in alphabetical order
                for residue in residues:
                    residue_idx = residue["res_idx"] + 1
                    self.qa_metrics.append(
                        _LocalPLDDT(
                            asym_unit_map[chain_idx].residue(residue_idx),
                            plddts[res_num].item(),
                        )
                    )
                    res_num += 1

    # Add the model and modeling protocol to the file and write them out:
    model = _MyModel(assembly=modeled_assembly, name="Model")
    if plddts is not None:
        model.add_plddt(plddts)

    model_group = ModelGroup([model], name="All models")
    system.model_groups.append(model_group)

    fh = io.StringIO()
    dumper.write(fh, [system])
    return fh.getvalue()


def save_structure(structure, save_dir, outname, output_format="mmcif", plddts=None):
    if output_format == "pdb":
        path = save_dir / f"{outname}.pdb"
        with path.open("w") as f:
            f.write(to_pdb(structure, plddts=plddts))
    elif output_format == "mmcif":
        path = save_dir / f"{outname}.cif"
        with path.open("w") as f:
            f.write(to_mmcif(structure, plddts=plddts))
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


class SimpleFoldConfig(PretrainedConfig):
    model_type = "simplefold"
    def __init__(
        self,
        # General
        ema_decay: float = 0.999,
        esm_model_name_or_path: str = "Synthyra/ESM2-3B",
        use_rigid_align: bool = True,
        smooth_lddt_loss_weight: float = 1.0,
        lddt_cutoff: float = 15.0,
        clip_grad_norm_val: T.Optional[float] = None,
        lddt_weight_schedule: bool = False,
        plddt_training: bool = False,
        sample_dir: str = 'artifacts/',

        # Residue trunk
        hidden_size: int = 1152,
        num_heads: int = 16,
        trunk_depth: int = 24,
        mlp_ratio: float = 4.0,
        use_swiglu: bool = True,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_use_bias: bool = True,
        rope_base: float = 100.0,

        # Atom encoder/decoder
        output_channels: int = 3,
        atom_hidden_size_enc: int = 256,
        atom_hidden_size_dec: int = 256,
        atom_num_heads: int = 4,
        encoder_depth: int = 8,
        decoder_depth: int = 8,
        atom_n_queries_enc: int = 32,
        atom_n_keys_enc: int = 128,
        atom_n_queries_dec: int = 32,
        atom_n_keys_dec: int = 128,

        # ESM integration
        esm_num_layers: int = 36,
        esm_embed_dim: int = 2560,
        esm_dropout_prob: float = 0.0,

        # Feature toggles
        use_atom_mask: bool = False,
        use_length_condition: bool = True,

        # Positional embeddings
        coord_pe_num_freqs: int = 32,
        coord_pe_min_freq_log2: float = 0.0,
        coord_pe_max_freq_log2: float = 12.0,
        coord_pe_include_input: bool = False,
        aa_pos_embed_dim: int = 64,
        time_embed_dim: int = 256,

        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ema_decay = ema_decay
        self.esm_model_name_or_path = esm_model_name_or_path
        self.use_rigid_align = use_rigid_align
        self.smooth_lddt_loss_weight = smooth_lddt_loss_weight
        self.lddt_cutoff = lddt_cutoff
        self.clip_grad_norm_val = clip_grad_norm_val
        self.lddt_weight_schedule = lddt_weight_schedule
        self.plddt_training = plddt_training
        self.sample_dir = sample_dir

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.trunk_depth = trunk_depth
        self.mlp_ratio = mlp_ratio
        self.use_swiglu = use_swiglu
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.attn_use_bias = attn_use_bias
        self.rope_base = rope_base

        self.output_channels = output_channels
        self.atom_hidden_size_enc = atom_hidden_size_enc
        self.atom_hidden_size_dec = atom_hidden_size_dec
        self.atom_num_heads = atom_num_heads
        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth
        self.atom_n_queries_enc = atom_n_queries_enc
        self.atom_n_keys_enc = atom_n_keys_enc
        self.atom_n_queries_dec = atom_n_queries_dec
        self.atom_n_keys_dec = atom_n_keys_dec

        self.esm_num_layers = esm_num_layers
        self.esm_embed_dim = esm_embed_dim
        self.esm_dropout_prob = esm_dropout_prob

        self.use_atom_mask = use_atom_mask
        self.use_length_condition = use_length_condition

        self.coord_pe_num_freqs = coord_pe_num_freqs
        self.coord_pe_min_freq_log2 = coord_pe_min_freq_log2
        self.coord_pe_max_freq_log2 = coord_pe_max_freq_log2
        self.coord_pe_include_input = coord_pe_include_input
        self.aa_pos_embed_dim = aa_pos_embed_dim
        self.time_embed_dim = time_embed_dim


class SimpleFold(PreTrainedModel, SimpleFoldUtils):
    config_class = SimpleFoldConfig
    def __init__(self, config: SimpleFoldConfig):
        PreTrainedModel.__init__(self, config)
        SimpleFoldUtils.__init__(self)
        self.config = config

        # Core training/inference settings
        self.use_rigid_align = config.use_rigid_align
        self.lddt_cutoff = config.lddt_cutoff
        self.smooth_lddt_loss_weight = config.smooth_lddt_loss_weight
        self.use_smooth_lddt_loss = config.smooth_lddt_loss_weight > 0.0
        self.lddt_weight_schedule = config.lddt_weight_schedule
        self.sample_dir = config.sample_dir
        self.nval_steps = 0
        self.t_eps = 0.0

        # Embedders
        self.pos_embedder = FourierPositionEncoding(
            in_dim=3,
            include_input=config.coord_pe_include_input,
            min_freq_log2=config.coord_pe_min_freq_log2,
            max_freq_log2=config.coord_pe_max_freq_log2,
            num_freqs=config.coord_pe_num_freqs,
            log_sampling=True,
        )
        pos_embed_channels = self.pos_embedder.embed_dim

        self.aminoacid_pos_embedder = AbsolutePositionEncoding(
            in_dim=1,
            embed_dim=config.aa_pos_embed_dim,
            include_input=False,
        )
        aminoacid_pos_embed_channels = self.aminoacid_pos_embedder.embed_dim

        self.time_embedder = TimestepEmbedder(
            hidden_size=config.hidden_size,
            frequency_embedding_size=config.time_embed_dim,
        )

        # Attention rotary embeddings for atoms and tokens
        self.atom_rope = AxialRotaryPositionEncoding(
            in_dim=4, embed_dim=config.atom_hidden_size_enc, num_heads=config.atom_num_heads, base=config.rope_base
        )
        self.token_rope = AxialRotaryPositionEncoding(
            in_dim=4, embed_dim=config.hidden_size, num_heads=config.num_heads, base=config.rope_base
        )

        # Transformer stacks
        self.atom_encoder_transformer = StackedDiT(
            cfg=config,
            hidden_size=config.atom_hidden_size_enc,
            depth=config.encoder_depth,
            num_heads_override=config.atom_num_heads,
            pos_embedder=self.atom_rope,
        )
        self.atom_decoder_transformer = StackedDiT(
            cfg=config,
            hidden_size=config.atom_hidden_size_dec,
            depth=config.decoder_depth,
            num_heads_override=config.atom_num_heads,
            pos_embedder=self.atom_rope,
        )
        self.trunk = StackedDiT(
            cfg=config,
            hidden_size=config.hidden_size,
            depth=config.trunk_depth,
            num_heads_override=config.num_heads,
            pos_embedder=self.token_rope,
        )

        # Projections and heads
        self.hidden_size = config.hidden_size
        self.output_channels = config.output_channels
        self.num_heads = config.num_heads
        self.atom_num_heads = config.atom_num_heads
        self.use_atom_mask = config.use_atom_mask
        self.esm_dropout_prob = config.esm_dropout_prob
        self.use_length_condition = config.use_length_condition

        self.atom_hidden_size_enc = config.atom_hidden_size_enc
        self.atom_hidden_size_dec = config.atom_hidden_size_dec
        self.atom_n_queries_enc = config.atom_n_queries_enc
        self.atom_n_keys_enc = config.atom_n_keys_enc
        self.atom_n_queries_dec = config.atom_n_queries_dec
        self.atom_n_keys_dec = config.atom_n_keys_dec

        atom_feat_dim = pos_embed_channels + aminoacid_pos_embed_channels + 427
        self.atom_feat_proj = nn.Sequential(
            nn.Linear(atom_feat_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.SiLU(),
        )
        self.atom_pos_proj = nn.Linear(pos_embed_channels, self.hidden_size, bias=False)

        if self.use_length_condition:
            self.length_embedder = nn.Sequential(
                nn.Linear(1, self.hidden_size, bias=False),
                nn.LayerNorm(self.hidden_size),
            )

        self.atom_in_proj = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)

        self.esm_s_combine = nn.Parameter(torch.zeros(config.esm_num_layers))
        self.esm_s_proj = ConditionEmbedder(
            input_dim=config.esm_embed_dim,
            hidden_size=self.hidden_size,
            dropout_prob=self.esm_dropout_prob,
        )
        latent_cat_dim = self.hidden_size * 2
        self.esm_cat_proj = nn.Linear(latent_cat_dim, self.hidden_size)

        self.context2atom_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.atom_hidden_size_enc),
            nn.LayerNorm(self.atom_hidden_size_enc),
        )
        self.atom2latent_proj = nn.Sequential(
            nn.Linear(self.atom_hidden_size_enc, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
        )
        self.atom_enc_cond_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.atom_hidden_size_enc),
            nn.LayerNorm(self.atom_hidden_size_enc),
        )
        self.atom_dec_cond_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.atom_hidden_size_dec),
            nn.LayerNorm(self.atom_hidden_size_dec),
        )

        self.latent2atom_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SiLU(),
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.atom_hidden_size_dec),
        )

        self.final_layer = FinalLayer(
            self.atom_hidden_size_dec,
            self.output_channels,
            c_dim=self.hidden_size,
        )

        # prepare data tokenizer, featurizer, and processor
        self.tokenizer = BoltzTokenizer()
        self.featurizer = BoltzFeaturizer()
        self.processor = ProteinDataProcessor(
            device=self.device,
            scale=16.0,
            ref_scale=5.0,
            multiplicity=1,
            inference_multiplicity=self.nsample_per_protein,
            backend=self.backend,
        )

        self.flow = LinearPath()
        self.sampler = EMSampler(
            num_timesteps=self.num_steps,
            t_start=1e-4,
            tau=self.tau,
            log_timesteps=True,
            w_cutoff=0.99,
        )

        # ESM HF model (optional, used if you call init_esm_model)
        self.esm_model = None
        self.esm_tokenizer = None

    def init_esm_model(self):
        path = self.config.esm_model_name_or_path
        self.esm_model = AutoModel.from_pretrained(path, trust_remote_code=True)
        # Some ESM HF models expose a tokenizer attribute, others require separate AutoTokenizer
        self.esm_tokenizer = getattr(self.esm_model, "tokenizer", None)

    def init_plddt_modules(self):
        # loads plddt modules from checkpoints
        pass

    def forward(self, noised_pos, t, feats, self_cond=None):
        B, N, _ = feats["ref_pos"].shape
        M = feats["mol_type"].shape[1]
        atom_to_token = feats["atom_to_token"].float() # [B, N, M]
        atom_to_token_idx = feats["atom_to_token_idx"]
        ref_space_uid = feats["ref_space_uid"]

        # create atom attention masks
        atom_attn_mask_enc = self.create_atom_attn_mask(
            feats, 
            natoms=N,
            atom_n_queries=self.atom_n_queries_enc,
            atom_n_keys=self.atom_n_keys_enc,
        )
        atom_attn_mask_dec = self.create_atom_attn_mask(
            feats,
            natoms=N,
            atom_n_queries=self.atom_n_queries_dec,
            atom_n_keys=self.atom_n_keys_dec,
        )

        # create condition embeddings for AdaLN
        c_emb = self.time_embedder(t)  # (B, D)
        if self.use_length_condition:
            length = feats["max_num_tokens"].float().unsqueeze(-1)
            c_emb = c_emb + self.length_embedder(torch.log(length))

        # create atom features
        mol_type = feats["mol_type"]
        mol_type = F.one_hot(mol_type, num_classes=4).float()       # [B, M, 4]
        res_type = feats["res_type"].float()                        # [B, M, 33]
        pocket_feature = feats["pocket_feature"].float()            # [B, M, 4]
        res_feat = torch.cat(
            [mol_type, res_type, pocket_feature], 
        dim=-1)                                                     # [B, M, 41]
        atom_feat_from_res = torch.bmm(atom_to_token, res_feat)     # [B, N, 41]
        atom_res_pos = self.aminoacid_pos_embedder(
            pos=atom_to_token_idx.unsqueeze(-1).float()
        )
        ref_pos_emb = self.pos_embedder(pos=feats["ref_pos"])
        atom_feat = torch.cat(
            [
                ref_pos_emb,                                        # (B, N, PD1)
                atom_feat_from_res,                                 # (B, N, 41)
                atom_res_pos,                                       # (B, N, PD2)
                feats["ref_charge"].unsqueeze(-1),                  # (B, N, 1)
                feats["atom_pad_mask"].unsqueeze(-1),               # (B, N, 1)
                feats["ref_element"],                               # (B, N, 128)
                feats["ref_atom_name_chars"].reshape(B, N, 4 * 64), # (B, N, 256)
            ],
            dim=-1,
        )                                                           # (B, N, PD1+PD2+427)
        atom_feat = self.atom_feat_proj(atom_feat)                  # (B, N, D)

        atom_coord = self.pos_embedder(pos=noised_pos)              # (B, N, PD1)
        atom_coord = self.atom_pos_proj(atom_coord)                 # (B, N, D)

        atom_in = torch.cat([atom_feat, atom_coord], dim=-1)
        atom_in = self.atom_in_proj(atom_in)                        # (B, N, D)

        # position embeddings for Axial RoPE
        atom_pe_pos = torch.cat(
            [
                ref_space_uid.unsqueeze(-1).float(),                 # (B, N, 1)
                feats["ref_pos"],                                    # (B, N, 3)
            ],
            dim=-1,
        )                                                            # (B, N, 4)
        token_pe_pos = torch.cat(
            [
                feats["residue_index"].unsqueeze(-1).float(),        # (B, M, 1)
                feats["entity_id"].unsqueeze(-1).float(),            # (B, M, 1)
                feats["asym_id"].unsqueeze(-1).float(),              # (B, M, 1)
                feats["sym_id"].unsqueeze(-1).float(),               # (B, M, 1)
            ],
            dim=-1,
        )                                                            # (B, M, 4)

        # atom encoder
        atom_c_emb_enc = self.atom_enc_cond_proj(c_emb)
        atom_latent = self.context2atom_proj(atom_in)
        atom_latent = self.atom_encoder_transformer(
            latents=atom_latent, 
            c=atom_c_emb_enc, 
            attention_mask=atom_attn_mask_enc,
            pos=atom_pe_pos,
        )
        atom_latent = self.atom2latent_proj(atom_latent)

        # grouping: aggregate atom tokens to residue tokens
        atom_to_token_mean = atom_to_token / (
            atom_to_token.sum(dim=1, keepdim=True) + 1e-6
        )
        latent = torch.bmm(atom_to_token_mean.transpose(1, 2), atom_latent)
        assert latent.shape[1] == M

        esm_s = (self.esm_s_combine.softmax(0).unsqueeze(0) @ feats['esm_s']).squeeze(2)
        force_drop_ids = feats.get("force_drop_ids", None)
        esm_emb = self.esm_s_proj(esm_s, self.training, force_drop_ids)
        assert esm_emb.shape[1] == latent.shape[1]

        latent = self.esm_cat_proj(torch.cat([latent, esm_emb], dim=-1))

        # residue trunk
        latent = self.trunk(
            latents=latent, 
            c=c_emb, 
            attention_mask=None,
            pos=token_pe_pos,
        )

        # ungrouping: broadcast residue tokens to atom tokens
        output = torch.bmm(atom_to_token, latent)
        assert output.shape[1] == N

        # add skip connection
        output = output + atom_latent
        output = self.latent2atom_proj(output)

        # atom decoder
        atom_c_emb_dec = self.atom_dec_cond_proj(c_emb)
        output = self.atom_decoder_transformer(
            latents=output, 
            c=atom_c_emb_dec,
            attention_mask=atom_attn_mask_dec,
            pos=atom_pe_pos,
        )
        output = self.final_layer(output, c=c_emb)

        return {
            "predict_velocity": output,
            "latent": latent,
        }

    def inference(self, aa_seq):
        self.eval()
        device = next(self.parameters()).device

        # Build a minimal synthetic Structure from sequence
        chains = aa_seq.split(":")
        residues = []
        atoms_list = []
        bonds = np.zeros((0,), dtype=Bond)
        connections = np.zeros((0,), dtype=Connection)
        interfaces = np.zeros((0,), dtype=Interface)
        mask = []

        atom_idx = 0
        res_idx = 0
        chain_array = []
        entity_id = 0
        for asym_id, chain_seq in enumerate(chains):
            chain_atom_idx = atom_idx
            chain_res_idx = res_idx
            for letter in chain_seq:
                res3 = prot_letter_to_token.get(letter, "UNK")
                atom_names = const.ref_atoms.get(res3, [])
                atom_num = len(atom_names)
                # Build Atom rows
                for an in atom_names:
                    name_arr = np.zeros((4,), dtype=np.int8)
                    for i, ch in enumerate(an[:4]):
                        name_arr[i] = (ord(ch) - 32)
                    atoms_list.append(
                        (name_arr, 0, 0, np.zeros((3,), dtype=np.float32), np.zeros((3,), dtype=np.float32), True, 0)
                    )
                # Residue row
                residues.append((res3, const.token_ids.get(res3, const.token_ids["UNK"]) , res_idx, atom_idx, atom_num, 0, 1 if atom_num>0 else 0, True, True))
                atom_idx += atom_num
                res_idx += 1
            # Chain row (mol_type protein)
            chain_array.append(("A", const.chain_type_ids["PROTEIN"], entity_id, 0, asym_id, chain_atom_idx, atom_idx - chain_atom_idx, chain_res_idx, res_idx - chain_res_idx))
            mask.append(True)
            entity_id += 1

        atoms_np = np.array(atoms_list, dtype=Atom) if len(atoms_list)>0 else np.zeros((0,), dtype=Atom)
        residues_np = np.array(residues, dtype=Residue) if len(residues)>0 else np.zeros((0,), dtype=Residue)
        chains_np = np.array(chain_array, dtype=Chain) if len(chain_array)>0 else np.zeros((0,), dtype=Chain)
        structure = Structure(
            atoms=atoms_np,
            bonds=bonds,
            residues=residues_np,
            chains=chains_np,
            connections=connections,
            interfaces=interfaces,
            mask=np.array(mask, dtype=bool) if len(mask)>0 else np.zeros((0,), dtype=bool),
        )

        # Tokenize and featurize
        tokenized = self.tokenizer.tokenize(Input(structure, {}))
        features = self.featurizer.process(tokenized)

        # Extract sequence back in chain-delimited form for ESM processing
        # Build from residues per chain
        seq_per_chain = []
        for ch in chains_np:
            start = ch[7]
            count = ch[8]
            letters = []
            for r in residues_np[start:start+count]:
                res3 = r[0]
                # map 3-letter back to 1-letter using common mapping
                back = {
                    "ALA":"A","ARG":"R","ASN":"N","ASP":"D",
                    "CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
                    "HIS":"H","ILE":"I","LEU":"L","LYS":"K",
                    "MET":"M","PHE":"F","PRO":"P","SER":"S",
                    "THR":"T","TRP":"W","TYR":"Y","VAL":"V",
                    "UNK":"X","-":"-"
                }.get(res3, "X")
                letters.append(back)
            seq_per_chain.append("".join(letters))
        sequence = ":".join(seq_per_chain)

        # Package batch
        features["aa_seq"] = [sequence]
        features["record"] = {}
        features["num_repeats"] = torch.tensor(1)
        features["max_num_tokens"] = torch.tensor(len(tokenized.tokens), dtype=torch.long)
        features["cropped_num_tokens"] = torch.tensor(len(tokenized.tokens), dtype=torch.long)

        # Simple in-file collate for single example
        batch = {k: (v if isinstance(v, torch.Tensor) else torch.tensor(v) if isinstance(v, (np.ndarray, np.generic)) else v) for k,v in features.items()}
        # Ensure tensor batch dims
        for k,v in list(batch.items()):
            if isinstance(v, torch.Tensor) and v.ndim>=1:
                batch[k] = v.unsqueeze(0)

        # Initialize ESM if available
        if self.esm_model is None:
            try:
                self.init_esm_model()
            except Exception:
                self.esm_model = None
                self.esm_tokenizer = None

        # Minimal esm_dict for compute_language_model_representations compatibility
        if self.esm_model is not None:
            class _ESMDict:
                def __init__(self):
                    self.cls_idx = 0
                    self.eos_idx = 2
                    self.padding_idx = 1
                def get_idx(self, x):
                    return 0
            self.esm_dict = _ESMDict()
            self.af2_to_esm = _af2_to_esm(self.esm_dict)
        else:
            self.esm_dict = None
            self.af2_to_esm = None

        # Preprocess to device and generate ESM features if possible
        batch = self.processor.preprocess_inference(
            batch,
            esm_model=self.esm_model,
            esm_dict=self.esm_dict,
            af2_to_esm=self.af2_to_esm,
        )

        # Ensure esm_s exists for model forward
        if "esm_s" not in batch:
            B = batch["res_type"].shape[0]
            L = batch["res_type"].shape[1]
            batch["esm_s"] = torch.zeros(
                (B, L, self.config.esm_num_layers + 1, self.config.esm_embed_dim),
                device=device,
            )

        # Sample
        noise = torch.randn_like(batch["coords"]).to(device)
        out_dict = self.sampler.sample(self, self.flow, noise, batch)

        # Optional pLDDT
        plddts = None
        if hasattr(self, "plddt_out_module") and hasattr(self, "plddt_latent_module"):
            if (self.plddt_out_module is not None) and (self.plddt_latent_module is not None):
                t = torch.ones(batch["coords"].shape[0], device=device)
                out_feat = self.plddt_latent_module(out_dict["denoised_coords"].detach(), t, batch)
                plddt_out_dict = self.plddt_out_module(out_feat["latent"].detach(), batch)
                plddts = plddt_out_dict["plddt"] * 100.0

        out_dict = self.processor.postprocess(out_dict, batch)
        sampled_coord = out_dict["denoised_coords"].detach()
        return sampled_coord, batch["atom_pad_mask"], plddts
