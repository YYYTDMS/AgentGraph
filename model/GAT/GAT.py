import torch.nn as nn
import torch.nn.functional as F
from cogdl.models import BaseModel
from cogdl.layers import GATLayer
class GAT(BaseModel):
    def __init__(
            self,
            in_feats,
            hidden_size,
            out_feats,
            num_layers,
            dropout=0.5,
            input_drop=0.25,
            attn_drop=0.0,
            alpha=0.2,
            nhead=4,
            residual=True,
            last_nhead=1,
            norm="batchnorm",
    ):
        """Sparse version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.input_drop = input_drop
        self.attentions = nn.ModuleList()
        self.attentions.append(
            GATLayer(in_feats, hidden_size, nhead=nhead, attn_drop=attn_drop, alpha=alpha, residual=residual, norm=norm)
        )
        for i in range(num_layers - 2):
            self.attentions.append(
                GATLayer(
                    hidden_size * nhead,
                    hidden_size,
                    nhead=nhead,
                    attn_drop=attn_drop,
                    alpha=alpha,
                    residual=residual,
                    norm=norm,
                )
            )
        self.attentions.append(
            GATLayer(
                hidden_size * nhead,
                out_feats,
                attn_drop=attn_drop,
                alpha=alpha,
                nhead=last_nhead,
                residual=False,
            )
        )
        self.num_layers = num_layers
        self.last_nhead = last_nhead
        self.residual = residual

    def forward(self, graph):
        graph.sym_norm()
        x = graph.x
        x = F.dropout(x, p=self.input_drop, training=self.training)
        for i, layer in enumerate(self.attentions):
            x = layer(graph, x)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x