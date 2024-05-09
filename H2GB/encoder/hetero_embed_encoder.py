import torch
import torch.nn as nn
from H2GB.graphgym.config import cfg
from H2GB.graphgym.register import (register_node_encoder,
                                               register_edge_encoder)


@register_node_encoder('Hetero_Embed')
class TypeDictNodeEncoder(torch.nn.Module):
    def __init__(self, dim_in, dim_emb, dataset):
        super().__init__()

        # self.encoder_dict = nn.ModuleDict()
        # self.encoder = torch.nn.Embedding(num_embeddings=num_types,
        #                                   embedding_dim=dim_emb)
        print('embedding_dim:', dim_emb)
        self.data = dataset[0]
        self.encoder_dict = nn.ModuleDict(
            {
                node_type: nn.Embedding(self.data[node_type].num_nodes, dim_emb)
                for node_type in self.data.node_types
                if not hasattr(self.data[node_type], 'x')
            }
        )
        # torch.nn.init.xavier_uniform_(self.encoder.weight.data)

    def forward(self, batch):
        # Encode just the first dimension if more exist
        for node_type, encoder in self.encoder_dict:
            if node_type in batch.node_types:
                batch[node_type].x = encoder(batch[node_type].n_id)

        return batch
