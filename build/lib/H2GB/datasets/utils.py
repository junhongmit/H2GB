import os
import os.path as osp
import gdown
import torch
import numpy as np
from torch_sparse import SparseTensor

def download_dataset(url, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    file_id = url.split('/')[-2]
    download_url = f'https://drive.google.com/uc?id={file_id}'
    
    # output_path = os.path.join(output_directory, 'dataset.zip')
    try:
        output_path = gdown.download(download_url, output_directory + osp.sep, quiet=False)
    except:
        print('It looks like Gdown encounters errors, or Google drive exhibits download '
              'number limits during downloading. However, You still can download the file '
              'from a web browser by using this link:\n\n'
              f'{url}\n\n'
              'Then unzip this file (only if it is a Zip file), and manually put all the content to '
              f'{output_directory}.')
        exit(0)

    
    # Unzip the dataset if necessary
    if output_path.endswith('.zip'):
        import zipfile
        with zipfile.ZipFile(output_path, 'r') as zip_ref:
            zip_ref.extractall(output_directory)
        os.remove(output_path)  # Remove the zip file after extraction
    

def even_quantile_labels(vals, nclasses, verbose=True):
    """ partitions vals into nclasses by a quantile based split,
    where the first class is less than the 1/nclasses quantile,
    second class is less than the 2/nclasses quantile, and so on
    
    vals is np array
    returns an np array of int class labels
    """
    label = -1 * np.ones(vals.shape[0], dtype=int)
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.nanquantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    if verbose:
        print('Class Label Intervals:')
        for class_idx, interval in enumerate(interval_lst):
            print(f'Class {class_idx}: [{interval[0]}, {interval[1]})]')
    return label

def get_sparse_tensor(edge_index, num_nodes=None, num_src_nodes=None, num_dst_nodes=None, return_e_id=False):
    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if return_e_id:
            value = torch.arange(adj_t.nnz())
            adj_t = adj_t.set_value(value, layout='coo')
        return adj_t


    if (num_nodes is None) and (num_src_nodes is None) and (num_dst_nodes is None):
        num_src_nodes = int(edge_index.max()) + 1
        num_dst_nodes = num_src_nodes
    elif (num_src_nodes is None) and (num_dst_nodes is None):
        num_src_nodes, num_dst_nodes = num_nodes, num_nodes


    value = torch.arange(edge_index.size(1)) if return_e_id else None
    return SparseTensor(row=edge_index[0], col=edge_index[1],
                        value=value,
                        sparse_sizes=(num_src_nodes, num_dst_nodes)).t()