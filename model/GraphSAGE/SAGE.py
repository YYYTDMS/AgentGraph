import torch.nn as nn
from cogdl.models import BaseModel
from cogdl.layers import SAGELayer
class SAGE(BaseModel):
    def __init__(
        self,
        in_feats,
        out_feats,
        hidden_size,
        num_layers,
        aggr="mean",
        dropout=0.5,
        norm="batchnorm",
        activation="relu",
        normalize=False,
    ):
        super(SAGE, self).__init__()
        shapes = [in_feats] + [hidden_size] * (num_layers - 1) + [out_feats]
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [
                SAGELayer(
                    shapes[i],
                    shapes[i + 1],
                    aggr=aggr,
                    normalize=normalize if i != num_layers - 1 else False,
                    dropout=dropout if i != num_layers - 1 else False,
                    norm=norm if i != num_layers - 1 else None,
                    activation=activation if i != num_layers - 1 else None,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, graph):
        graph.sym_norm()
        x = graph.x
        for layer in self.layers:
            x = layer(graph, x)
        return x