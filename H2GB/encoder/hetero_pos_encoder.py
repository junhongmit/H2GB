import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
import H2GB.graphgym.register as register
from H2GB.graphgym.config import cfg
from H2GB.graphgym.register import register_node_encoder
from H2GB.transform.posenc_stats import get_rw_landing_probs


class HeteroPENodeEncoder(torch.nn.Module):
    """Configurable Positional Encoding node encoder.

    PE of size `dim_pe` will get appended to each node feature vector.
    If `reshape_x` set True, original node features will be first linearly
    projected to (dim_emb - dim_pe) size and the concatenated with PE.

    Args:
        dim_emb: Size of final node embedding
        reshape_x: Reshape node features `x` from dim_in to (dim_emb - dim_pe)
    """

    kernel_type = None  # Instantiated type of the KernelPE, e.g. RWSE

    def __init__(self, dim_emb, dataset, reshape_x=True):
        super().__init__()
        self.is_hetero = isinstance(dataset[0], HeteroData)
        if self.kernel_type is None:
            raise ValueError(f"{self.__class__.__name__} has to be "
                             f"preconfigured by setting 'kernel_type' class"
                             f"variable before calling the constructor.")

        # dim_in = cfg.share.dim_in  # Expected original input node features dim

        pecfg = getattr(cfg, f"posenc_{self.kernel_type}")
        dim_pe = pecfg.dim_pe  # Size of the kernel-based PE embedding
        model_type = pecfg.model.lower()  # Encoder NN model type for PEs
        n_layers = pecfg.layers  # Num. layers in PE encoder model
        norm_type = pecfg.raw_norm_type.lower()  # Raw PE normalization layer type
        self.pass_as_var = pecfg.pass_as_var  # Pass PE also as a separate variable

        if dim_emb - dim_pe < 0:
            raise ValueError(f"PE dim size {dim_pe} is too large for "
                             f"desired embedding size of {dim_emb}.")

        if reshape_x and dim_emb - dim_pe > 0:
            dim_in = cfg.share.dim_in  # Expected original input node features dim
            if self.is_hetero:
                self.linear_x = nn.ModuleDict()
                for node_type in dim_in:
                    self.linear_x[node_type] = nn.Linear(dim_in[node_type], dim_emb - dim_pe)
            else:
                self.linear_x = nn.Linear(dim_in, dim_emb - dim_pe)
        self.reshape_x = (reshape_x and dim_emb - dim_pe > 0)

        activation = register.act_dict[cfg.gnn.act]
        if self.kernel_type == 'Hetero_RWSE':
            num_rw_steps = len(pecfg.kernel.times)
            if norm_type == 'batchnorm':
                self.raw_norm = nn.BatchNorm1d(num_rw_steps)
            else:
                self.raw_norm = None

            if model_type == 'mlp':
                layers = []
                if n_layers == 1:
                    layers.append(nn.Linear(num_rw_steps, dim_pe))
                    layers.append(activation())
                else:
                    layers.append(nn.Linear(num_rw_steps, 2 * dim_pe))
                    layers.append(activation())
                    for _ in range(n_layers - 2):
                        layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
                        layers.append(activation())
                    layers.append(nn.Linear(2 * dim_pe, dim_pe))
                    layers.append(activation())
                self.pe_encoder = nn.Sequential(*layers)
            elif model_type == 'linear':
                self.pe_encoder = nn.Linear(num_rw_steps, dim_pe)
            else:
                raise ValueError(f"{self.__class__.__name__}: Does not support "
                                f"'{model_type}' encoder model.")
        else:
            if model_type == 'mlp':
                layers = []
                if n_layers == 1:
                    layers.append(nn.Linear(dim_pe, dim_pe))
                    layers.append(activation())
                else:
                    layers.append(nn.Linear(dim_pe, 2 * dim_pe))
                    layers.append(activation())
                    for _ in range(n_layers - 2):
                        layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
                        layers.append(activation())
                    layers.append(nn.Linear(2 * dim_pe, dim_pe))
                    layers.append(activation())
                self.pe_encoder = nn.Sequential(*layers)
            elif model_type == 'linear':
                self.pe_encoder = nn.Linear(dim_pe, dim_pe)

    def forward(self, batch):
        pestat_var = f"pestat_{self.kernel_type}"
        if self.kernel_type in ['Hetero_Metapath', 'Hetero_Node2Vec', 'Hetero_TransE', 'Hetero_ComplEx']:
            if isinstance(batch, HeteroData):
                pos_enc = {node_type: batch[node_type][pestat_var] if hasattr(batch[node_type], pestat_var) else None for node_type in batch.node_types}
            else:
                pos_enc = batch[pestat_var]
        else:
            if not hasattr(batch, pestat_var):
                # raise ValueError(f"Precomputed '{pestat_var}' variable is "
                #                  f"required for {self.__class__.__name__}; set "
                #                  f"config 'posenc_{self.kernel_type}.enable' to "
                #                  f"True, and also set 'posenc.kernel.times' values")
                kernel_param = cfg.posenc_Hetero_RWSE.kernel

                homo_data = batch.to_homogeneous()
                edge_index = homo_data.edge_index
                rw_landing = get_rw_landing_probs(ksteps=kernel_param.times,
                                            edge_index=homo_data.edge_index,
                                            num_nodes=homo_data.node_type.numel())
                
                if self.raw_norm:
                    rw_landing = self.raw_norm(rw_landing)
                rw_landing = self.pe_encoder(rw_landing)  # (Num nodes) x dim_pe
                
                # To get back the original node type: iterate batch.num_nodes_dict
                pos_enc = {}
                for idx, (k, v) in enumerate(batch.num_nodes_dict.items()):
                    pos_enc[k] = rw_landing[homo_data.node_type == idx]
            else:
                pos_enc = getattr(batch, pestat_var)  # (Num nodes) x (Num kernel times)

                # pos_enc = batch.rw_landing  # (Num nodes) x (Num kernel times)
                if self.raw_norm:
                    pos_enc = self.raw_norm(pos_enc)
                pos_enc = self.pe_encoder(pos_enc)  # (Num nodes) x dim_pe

        # Expand node features if needed
        if isinstance(batch, HeteroData):
            for node_type in batch.x_dict:
                if hasattr(batch[node_type], pestat_var):
                    if self.reshape_x:
                        h = self.linear_x[node_type](batch.x_dict[node_type])
                    else:
                        h = batch.x_dict[node_type]
                    # Concatenate final PEs to input embedding
                    batch[node_type].x = torch.cat((h, pos_enc[node_type]), dim=-1)
        else:
            if self.reshape_x:
                h = self.linear_x(batch.x)
            else:
                h = batch.x
            # Concatenate final PEs to input embedding
            batch.x = torch.cat((h, pos_enc), dim=-1)

        # Keep PE also separate in a variable (e.g. for skip connections to input)
        if self.pass_as_var:
            setattr(batch, f'pe_{self.kernel_type}', pos_enc)
        return batch


@register_node_encoder('Hetero_RWSE')
class RWSENodeEncoder(HeteroPENodeEncoder):
    """Random Walk Structural Encoding node encoder.
    """
    kernel_type = 'Hetero_RWSE'


@register_node_encoder('Hetero_Metapath')
class MetapathNodeEncoder(HeteroPENodeEncoder):
    """Metapath2Vec based Structural Encoding node encoder.
    """
    kernel_type = 'Hetero_Metapath'

@register_node_encoder('Hetero_Node2Vec')
class Node2VecNodeEncoder(HeteroPENodeEncoder):
    """Node2Vec based Structural Encoding node encoder.
    """
    kernel_type = 'Hetero_Node2Vec'

@register_node_encoder('Hetero_TransE')
class TransENodeEncoder(HeteroPENodeEncoder):
    """Random Walk Structural Encoding node encoder.
    """
    kernel_type = 'Hetero_TransE'

@register_node_encoder('Hetero_ComplEx')
class ComplExNodeEncoder(HeteroPENodeEncoder):
    """Random Walk Structural Encoding node encoder.
    """
    kernel_type = 'Hetero_ComplEx'

@register_node_encoder('Hetero_DistMult')
class DistMultNodeEncoder(HeteroPENodeEncoder):
    """Random Walk Structural Encoding node encoder.
    """
    kernel_type = 'Hetero_DistMult'

