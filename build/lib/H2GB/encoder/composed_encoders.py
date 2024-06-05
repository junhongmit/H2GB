import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from H2GB.graphgym.config import cfg
from H2GB.graphgym.register import register_node_encoder

from H2GB.encoder.raw_encoder import (RawNodeEncoder, RawEdgeEncoder)
from H2GB.encoder.hetero_raw_encoder import (HeteroRawNodeEncoder, HeteroRawEdgeEncoder)
from H2GB.encoder.voc_superpixels_encoder import VOCNodeEncoder

from H2GB.encoder.laplace_pos_encoder import LapPENodeEncoder
from H2GB.encoder.homo_gnn_encoder import HomoGNNEncoder
from H2GB.encoder.hetero_gnn_encoder import HeteroGNNEncoder
from H2GB.encoder.hetero_label_encoder import HeteroLabelNodeEncoder
from H2GB.encoder.hetero_pos_encoder import (RWSENodeEncoder, Node2VecNodeEncoder, \
                                                  MetapathNodeEncoder, TransENodeEncoder, \
                                                  ComplExNodeEncoder, DistMultNodeEncoder)
from H2GB.encoder.hetero_distance_encoder import (HeteroDistanceAttentionBias, \
                                                       kHopAttentionBias, kHopAugmentation,\
                                                       HeteroDistancePositionEncoding)


def concat_node_encoders(encoder_classes, pe_enc_names):
    """
    A factory that creates a new Encoder class that concatenates functionality
    of the given list of two or three Encoder classes. First Encoder is expected
    to be a dataset-specific encoder, and the rest PE Encoders.

    Args:
        encoder_classes: List of node encoder classes
        pe_enc_names: List of PE embedding Encoder names, used to query a dict
            with their desired PE embedding dims. That dict can only be created
            during the runtime, once the config is loaded.

    Returns:
        new node encoder class
    """

    class Concat2NodeEncoder(torch.nn.Module):
        """Encoder that concatenates two node encoders.
        """
        enc1_cls = None
        enc2_cls = None
        enc2_name = None

        def __init__(self, dim_emb, data):
            super().__init__()
            self.is_hetero = isinstance(data, HeteroData)
            
            if cfg.posenc_EquivStableLapPE.enable: # Special handling for Equiv_Stable LapPE where node feats and PE are not concat
                self.encoder1 = self.enc1_cls(dim_emb, data)
                self.encoder2 = self.enc2_cls(dim_emb, data)
            else:
                # PE dims can only be gathered once the cfg is loaded.
                enc2_dim_pe = getattr(cfg, f"posenc_{self.enc2_name}").dim_pe

                # Concatenation
                self.encoder1 = self.enc1_cls(dim_emb - enc2_dim_pe, data)
                self.encoder2 = self.enc2_cls(dim_emb, data, reshape_x=False)
                # if dim_emb != enc2_dim_pe:
                #     # Concatenation
                #     self.encoder1 = self.enc1_cls(dim_emb - enc2_dim_pe, data)
                #     self.encoder2 = self.enc2_cls(dim_emb, data, reshape_x=False)
                # else:
                #     # Element-wise addition
                #     self.encoder1 = self.enc1_cls(dim_emb, data)
                #     self.encoder2 = self.enc2_cls(dim_emb, data, reshape_x=False)

                # if self.is_hetero:
                #     self.linear = nn.ModuleDict()
                #     for node_type in data.metadata()[0]:
                #             self.linear[node_type] = nn.Linear(
                #                 dim_in[node_type] + enc2_dim_pe, dim_emb
                #             )
                # else:
                #     self.linear = nn.Linear(
                #                     dim_in + enc2_dim_pe, dim_emb
                #                 )

        def forward(self, batch):
            batch = self.encoder1(batch)
            batch = self.encoder2(batch)
            # if self.is_hetero:
            #     for node_type, x in batch.x_dict.items():
            #         batch[node_type].x = self.linear[node_type](x) 
            # else:
            #     batch.x = self.linear(batch.x)
            return batch

    class Concat3NodeEncoder(torch.nn.Module):
        """Encoder that concatenates three node encoders.
        """
        enc1_cls = None
        enc2_cls = None
        enc2_name = None
        enc3_cls = None
        enc3_name = None

        def __init__(self, dim_emb, data):
            super().__init__()
            # PE dims can only be gathered once the cfg is loaded.
            enc2_dim_pe = getattr(cfg, f"posenc_{self.enc2_name}").dim_pe
            enc3_dim_pe = getattr(cfg, f"posenc_{self.enc3_name}").dim_pe
            self.encoder1 = self.enc1_cls(dim_emb - enc2_dim_pe - enc3_dim_pe, data)
            self.encoder2 = self.enc2_cls(dim_emb - enc3_dim_pe, data, reshape_x=False)
            self.encoder3 = self.enc3_cls(dim_emb, data, reshape_x=False)

        def forward(self, batch):
            batch = self.encoder1(batch)
            batch = self.encoder2(batch)
            batch = self.encoder3(batch)
            return batch

    # Configure the correct concatenation class and return it.
    if len(encoder_classes) == 2:
        Concat2NodeEncoder.enc1_cls = encoder_classes[0]
        Concat2NodeEncoder.enc2_cls = encoder_classes[1]
        Concat2NodeEncoder.enc2_name = pe_enc_names[0]
        return Concat2NodeEncoder
    elif len(encoder_classes) == 3:
        Concat3NodeEncoder.enc1_cls = encoder_classes[0]
        Concat3NodeEncoder.enc2_cls = encoder_classes[1]
        Concat3NodeEncoder.enc3_cls = encoder_classes[2]
        Concat3NodeEncoder.enc2_name = pe_enc_names[0]
        Concat3NodeEncoder.enc3_name = pe_enc_names[1]
        return Concat3NodeEncoder
    else:
        raise ValueError(f"Does not support concatenation of "
                         f"{len(encoder_classes)} encoder classes.")


# Dataset-specific node encoders.
ds_encs = {'Raw': RawNodeEncoder,
           'Hetero_Raw': HeteroRawNodeEncoder,
           'VOCNode': VOCNodeEncoder,}

# Positional Encoding node encoders.
pe_encs = {'LapPE': LapPENodeEncoder,
           'Homo_GNN': HomoGNNEncoder,
           'Hetero_RWSE': RWSENodeEncoder,
           'Hetero_Label': HeteroLabelNodeEncoder,
           'Hetero_Node2Vec': Node2VecNodeEncoder,
           'Hetero_Metapath': MetapathNodeEncoder,
           'Hetero_TransE': TransENodeEncoder,
           'Hetero_ComplEx': ComplExNodeEncoder,
           'Hetero_GNN': HeteroGNNEncoder,
           'Hetero_SDAB': HeteroDistanceAttentionBias,
           'Hetero_kHopAB': kHopAttentionBias,
           'Hetero_kHopAug': kHopAugmentation,
           'Hetero_SDPE': HeteroDistancePositionEncoding}

# Concat dataset-specific and PE encoders.
for ds_enc_name, ds_enc_cls in ds_encs.items():
    for pe_enc_name, pe_enc_cls in pe_encs.items():
        register_node_encoder(
            f"{ds_enc_name}+{pe_enc_name}",
            concat_node_encoders([ds_enc_cls, pe_enc_cls],
                                 [pe_enc_name])
        )

# Combine both Metapath and GNN structural encodings.
for ds_enc_name, ds_enc_cls in ds_encs.items():
    register_node_encoder(
        f"{ds_enc_name}+Hetero_Metapath+Hetero_GNN",
        concat_node_encoders([ds_enc_cls, MetapathNodeEncoder, HeteroGNNEncoder],
                             ['Hetero_Metapath', 'Hetero_GNN'])
    )

# Combine both Metapath and Label propagation encodings.
for ds_enc_name, ds_enc_cls in ds_encs.items():
    register_node_encoder(
        f"{ds_enc_name}+Hetero_Label+Hetero_Metapath",
        concat_node_encoders([ds_enc_cls, HeteroLabelNodeEncoder, MetapathNodeEncoder],
                             ['Hetero_Label', 'Hetero_Metapath'])
    )

for ds_enc_name, ds_enc_cls in ds_encs.items():
    register_node_encoder(
        f"{ds_enc_name}+Hetero_Label+Hetero_Node2Vec",
        concat_node_encoders([ds_enc_cls, HeteroLabelNodeEncoder, Node2VecNodeEncoder],
                             ['Hetero_Label', 'Hetero_Node2Vec'])
    )

# # Combine both LapPE and RWSE positional encodings.
# for ds_enc_name, ds_enc_cls in ds_encs.items():
#     register_node_encoder(
#         f"{ds_enc_name}+LapPE+RWSE",
#         concat_node_encoders([ds_enc_cls, LapPENodeEncoder, RWSENodeEncoder],
#                              ['LapPE', 'RWSE'])
#     )

# # Combine both SignNet and RWSE positional encodings.
# for ds_enc_name, ds_enc_cls in ds_encs.items():
#     register_node_encoder(
#         f"{ds_enc_name}+SignNet+RWSE",
#         concat_node_encoders([ds_enc_cls, SignNetNodeEncoder, RWSENodeEncoder],
#                              ['SignNet', 'RWSE'])
#     )

# # Combine GraphormerBias with LapPE or RWSE positional encodings.
# for ds_enc_name, ds_enc_cls in ds_encs.items():
#     register_node_encoder(
#         f"{ds_enc_name}+GraphormerBias+LapPE",
#         concat_node_encoders([ds_enc_cls, GraphormerEncoder, LapPENodeEncoder],
#                              ['GraphormerBias', 'LapPE'])
#     )
#     register_node_encoder(
#         f"{ds_enc_name}+GraphormerBias+RWSE",
#         concat_node_encoders([ds_enc_cls, GraphormerEncoder, RWSENodeEncoder],
#                              ['GraphormerBias', 'RWSE'])
#     )
