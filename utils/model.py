

from __future__ import annotations

import torch
from torch import nn
from torch_scatter import scatter_add, scatter_softmax

from utils.pair_mlp import PairMLP
from utils.torchdrug_env import setup_torchdrug_env

setup_torchdrug_env()

from torchdrug import data, models  # noqa: E402


class DSCrisk(nn.Module):

    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        pair_hidden: int,
        dropout: float,
        ddi_graph: dict,
        proj_dim: int = 128,
    ):
        super().__init__()
        self.num_drugs = int(ddi_graph["num_drugs"])
        # Keep edge list on CPU to avoid illegal GPU Graph construction after .to(cuda)
        edge_src = torch.tensor(ddi_graph["edge_src"], dtype=torch.long)
        edge_dst = torch.tensor(ddi_graph["edge_dst"], dtype=torch.long)
        self._edge_list_cpu = torch.stack([edge_src, edge_dst], dim=-1).contiguous()

        self.mol_encoder = models.GIN(
            input_dim=atom_dim,
            hidden_dims=[hidden_dim] * num_layers,
            edge_input_dim=bond_dim,
            batch_norm=True,
            short_cut=True,
            concat_hidden=False,
            readout="mean",
        )
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.drug_emb = nn.Embedding(self.num_drugs, hidden_dim)
        self.ddi_encoder = models.GCN(
            input_dim=hidden_dim,
            hidden_dims=[hidden_dim, hidden_dim],
            batch_norm=True,
            short_cut=True,
            concat_hidden=False,
            readout="mean",
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.pair_head = PairMLP(
            hidden_dim, num_classes, pair_hidden, dropout, proj_dim=proj_dim
        )
        self.hidden_dim = hidden_dim
        self._ddi_graph = None

    def _get_ddi_graph(self, device):
        device = torch.device(device)
        if self._ddi_graph is None or torch.device(self._ddi_graph.device) != device:
            graph = data.Graph(edge_list=self._edge_list_cpu, num_node=self.num_drugs)
            self._ddi_graph = graph.to(device)
        return self._ddi_graph

    def _encode_nodes(self, graph):
        out = self.mol_encoder(graph, graph.node_feature.float())
        return out["node_feature"], out["graph_feature"]

    def _cross_readout(self, node_feat, graph, partner_graph_feat):
        ctx = partner_graph_feat[graph.node2graph]
        score = (self.query(node_feat) * self.key(ctx)).sum(dim=-1, keepdim=True)
        score = score / (self.hidden_dim ** 0.5)
        attn = scatter_softmax(score, graph.node2graph, dim=0)
        value = self.value(node_feat)
        return scatter_add(attn * value, graph.node2graph, dim=0, dim_size=graph.batch_size)

    def encode_ddi(self, device):
        graph = self._get_ddi_graph(device)
        x = self.drug_emb.weight
        return self.ddi_encoder(graph, x)["node_feature"]

    def forward(self, graph_a, graph_b, idx_a=None, idx_b=None):
        if idx_a is None or idx_b is None:
            raise ValueError("drug_idx_a / drug_idx_b are required")

        node_a, graph_a_feat = self._encode_nodes(graph_a)
        node_b, graph_b_feat = self._encode_nodes(graph_b)
        inter_a = self._cross_readout(node_a, graph_a, graph_b_feat)
        inter_b = self._cross_readout(node_b, graph_b, graph_a_feat)

        ddi_all = self.encode_ddi(inter_a.device)
        ddi_a = ddi_all[idx_a]
        ddi_b = ddi_all[idx_b]

        h_a = self.fuse(torch.cat([inter_a, ddi_a], dim=-1))
        h_b = self.fuse(torch.cat([inter_b, ddi_b], dim=-1))
        return self.pair_head(h_a, h_b)


def load_DSCrisk(ckpt_path, device=None):

    ATOM_DIM = 67
    BOND_DIM = 18
    HIDDEN_DIM = 128
    GIN_LAYERS = 3
    NUM_CLASSES = 4
    PAIR_HIDDEN = 256
    DROPOUT = 0.2
    PROJ_DIM = 128

    if device is None:
        device = torch.device("cpu")
    else:
        device = torch.device(device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if set(ckpt.keys()) != {"model_state", "ddi_graph"}:
        raise KeyError(
            "ckpt.pt must contain only model_state and ddi_graph, "
            f"got keys={sorted(ckpt.keys())}"
        )

    model = DSCrisk(
        atom_dim=ATOM_DIM,
        bond_dim=BOND_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=GIN_LAYERS,
        num_classes=NUM_CLASSES,
        pair_hidden=PAIR_HIDDEN,
        dropout=DROPOUT,
        ddi_graph=ckpt["ddi_graph"],
        proj_dim=PROJ_DIM,
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model
