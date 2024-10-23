import torch.nn as nn
from cogdl.models import BaseModel
from cogdl.layers import GCNLayer
class GCN(BaseModel):
    def __init__(
            self,
            in_feats,
            hidden_size,
            out_feats,
            num_layers,
            dropout,
            activation="relu",
            norm="batchnorm",
    ):
        super(GCN, self).__init__()
        shapes = [in_feats] + [hidden_size] * (num_layers - 1) + [out_feats]
        self.layers = nn.ModuleList(
            [
                GCNLayer(
                    shapes[i],
                    shapes[i + 1],
                    dropout=dropout if i != num_layers - 1 else 0,
                    norm=norm if i != num_layers - 1 else None,
                    activation=activation if i != num_layers - 1 else None,
                )
                for i in range(num_layers)
            ]
        )
        self.num_layers = num_layers

    def forward(self, graph):
        graph.sym_norm()
        h = graph.x
        for i in range(self.num_layers):
            h = self.layers[i](graph, h)
        return h