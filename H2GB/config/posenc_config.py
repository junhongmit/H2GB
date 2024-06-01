from H2GB.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('posenc')
def set_cfg_posenc(cfg):
    """Extend configuration with positional encoding options.
    """

    # Argument group for each Positional Encoding class.
    cfg.posenc_LapPE = CN()
    cfg.posenc_SignNet = CN()
    cfg.posenc_RWSE = CN()
    cfg.posenc_Homo_GNN = CN()
    cfg.posenc_Hetero_RWSE = CN()
    cfg.posenc_Hetero_Label = CN()
    cfg.posenc_Hetero_Metapath = CN()
    cfg.posenc_Hetero_Node2Vec = CN()
    cfg.posenc_Hetero_TransE = CN()
    cfg.posenc_Hetero_ComplEx = CN()
    cfg.posenc_Hetero_DistMult = CN()
    cfg.posenc_Hetero_GNN = CN()
    cfg.posenc_HKdiagSE = CN()
    cfg.posenc_ElstaticSE = CN()
    cfg.posenc_EquivStableLapPE = CN()

    # Common arguments to all PE types.
    for name in ['posenc_LapPE', 'posenc_SignNet', 'posenc_RWSE', 'posenc_Hetero_RWSE',
                 'posenc_Homo_GNN', 'posenc_Hetero_Label',
                 'posenc_Hetero_Metapath', 'posenc_Hetero_Node2Vec', 'posenc_Hetero_TransE',
                 'posenc_Hetero_ComplEx', 'posenc_Hetero_DistMult', 'posenc_Hetero_GNN',
                 'posenc_HKdiagSE', 'posenc_ElstaticSE']:
        pecfg = getattr(cfg, name)

        # Use extended positional encodings
        pecfg.enable = False

        # Neural-net model type within the PE encoder:
        # 'DeepSet', 'Transformer', 'Linear', 'none', ...
        pecfg.model = 'none'

        # Size of Positional Encoding embedding
        pecfg.dim_pe = 16

        # Number of layers in PE encoder model
        pecfg.layers = 3

        # Number of attention heads in PE encoder when model == 'Transformer'
        pecfg.n_heads = 4

        # Number of layers to apply in LapPE encoder post its pooling stage
        pecfg.post_layers = 0

        # Choice of normalization applied to raw PE stats: 'none', 'BatchNorm'
        pecfg.raw_norm_type = 'none'

        # In addition to appending PE to the node features, pass them also as
        # a separate variable in the PyG graph batch object.
        pecfg.pass_as_var = False

    # Config for EquivStable LapPE
    cfg.posenc_EquivStableLapPE.enable = False
    cfg.posenc_EquivStableLapPE.raw_norm_type = 'none'

    # Config for Laplacian Eigen-decomposition for PEs that use it.
    for name in ['posenc_LapPE', 'posenc_SignNet', 'posenc_EquivStableLapPE']:
        pecfg = getattr(cfg, name)
        pecfg.eigen = CN()

        # The normalization scheme for the graph Laplacian: 'none', 'sym', or 'rw'
        pecfg.eigen.laplacian_norm = 'sym'

        # The normalization scheme for the eigen vectors of the Laplacian
        pecfg.eigen.eigvec_norm = 'L2'

        # Maximum number of top smallest frequencies & eigenvectors to use
        pecfg.eigen.max_freqs = 10

    # Config for SignNet-specific options.
    cfg.posenc_SignNet.phi_out_dim = 4
    cfg.posenc_SignNet.phi_hidden_dim = 64

    for name in ['posenc_RWSE', 'posenc_Hetero_RWSE', 'posenc_HKdiagSE', 'posenc_ElstaticSE']:
        pecfg = getattr(cfg, name)

        # Config for Kernel-based PE specific options.
        pecfg.kernel = CN()

        # List of times to compute the heat kernel for (the time is equivalent to
        # the variance of the kernel) / the number of steps for random walk kernel
        # Can be overridden by `posenc.kernel.times_func`
        pecfg.kernel.times = []

        # Python snippet to generate `posenc.kernel.times`, e.g. 'range(1, 17)'
        # If set, it will be executed via `eval()` and override posenc.kernel.times
        pecfg.kernel.times_func = ''

    # Override default, electrostatic kernel has fixed set of 10 measures.
    cfg.posenc_ElstaticSE.kernel.times_func = 'range(10)'

    cfg.posenc_Hetero_SDAB = CN()
    cfg.posenc_Hetero_SDAB.enable = False
    cfg.posenc_Hetero_SDAB.node_degrees_only = False
    cfg.posenc_Hetero_SDAB.dim_pe = 0
    cfg.posenc_Hetero_SDAB.num_spatial_types = 0
    cfg.posenc_Hetero_SDAB.enable_path = False
    # cfg.posenc_Hetero_SDAB.num_in_degrees = None
    # cfg.posenc_Hetero_SDAB.num_out_degrees = None

    cfg.posenc_Hetero_kHopAB = CN()
    cfg.posenc_Hetero_kHopAB.enable = False
    cfg.posenc_Hetero_kHopAB.num_spatial_types = 0
    cfg.posenc_Hetero_kHopAB.dim_pe = 0

    cfg.posenc_Hetero_kHopAug = CN()
    cfg.posenc_Hetero_kHopAug.enable = False
    cfg.posenc_Hetero_kHopAug.num_spatial_types = 0
    cfg.posenc_Hetero_kHopAug.dim_pe = 0

    cfg.posenc_Hetero_SDPE = CN()
    cfg.posenc_Hetero_SDPE.enable = False
    cfg.posenc_Hetero_SDPE.num_spatial_types = 0
    cfg.posenc_Hetero_SDPE.dim_pe = 0

    cfg.posenc_Homo_GNN.pre_layers = 0
    cfg.posenc_Homo_GNN.batch_norm = False
    cfg.posenc_Homo_GNN.layer_norm = False
    cfg.posenc_Homo_GNN.input_dropout = 0.0
    cfg.posenc_Homo_GNN.attn_dropout = 0.0
    cfg.posenc_Homo_GNN.dropout = 0.0
    cfg.posenc_Homo_GNN.act = 'relu'
    cfg.posenc_Homo_GNN.agg = 'mean'

    cfg.posenc_Hetero_GNN.pre_layers = 0
    cfg.posenc_Hetero_GNN.batch_norm = False
    cfg.posenc_Hetero_GNN.layer_norm = False
    cfg.posenc_Hetero_GNN.input_dropout = 0.0
    cfg.posenc_Hetero_GNN.attn_dropout = 0.0
    cfg.posenc_Hetero_GNN.dropout = 0.0
    cfg.posenc_Hetero_GNN.act = 'relu'
    cfg.posenc_Hetero_GNN.agg = 'mean'