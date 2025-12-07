import torch
from torch.utils.data import Dataset
import tiledb
import numpy as np
from scipy.sparse import coo_matrix

class TileDBCollator:
    def __init__(self, tiledb_path: str, n_genes: int, ctx_cfg: dict = None):
        self.tiledb_path = tiledb_path
        self.n_genes = n_genes
        # Default config optimized for concurrent reading without file locks
        self.ctx_cfg = ctx_cfg if ctx_cfg else {
            "sm.compute_concurrency_level": "1",
            "sm.io_concurrency_level": "1",
            "vfs.file.enable_filelocks": "false",
        }
        self._ctx = None
        self._array = None

    def _get_array(self):
        if self._ctx is None:
            self._ctx = tiledb.Ctx(self.ctx_cfg)
        if self._array is None:
            self._array = tiledb.open(self.tiledb_path, mode="r", ctx=self._ctx)
        return self._array

    def __call__(self, batch_indices):
        # batch_indices is a list of integers from the dataset
        if not batch_indices:
            return torch.empty(0, self.n_genes), torch.empty(0)
            
        # Ensure indices are a list/array
        indices = batch_indices
        
        A = self._get_array()
        
        # Use multi_index for batched reading - this is the key optimization
        # It fetches all data for the requested cells in one C++ call
        try:
            results = A.multi_index[indices, :]
        except Exception as e:
            # Attempt to re-open if there's an issue (e.g. context/handle invalid)
            # This is a basic recovery mechanism
            self._array = tiledb.open(self.tiledb_path, mode="r", ctx=self._ctx)
            results = self._array.multi_index[indices, :]

        # Identify dimensions and attributes
        # Based on user's previous code, we assume generic handling or specific names.
        # We'll dynamically get dimension names to be safe, but fallback to "gene_index" if needed.
        dom = A.schema.domain
        dim0_name = dom.dim(0).name # Cell dimension
        dim1_name = dom.dim(1).name # Gene dimension
        attr_name = A.schema.attr(0).name # Data attribute
        
        cell_coords = results[dim0_name]
        gene_coords = results[dim1_name]
        data_vals = results[attr_name]
        
        # We need to map global cell indices back to the batch row index (0..batch_size-1)
        # Create a mapping from global_index -> batch_row_index
        # Note: 'indices' passed to this function matches the order of the batch
        idx_map = {global_idx: i for i, global_idx in enumerate(indices)}
        
        # Vectorized mapping might be faster but this is robust
        row_indices = np.array([idx_map[c] for c in cell_coords])
        
        # Construct sparse matrix then densify
        # shape: (batch_size, n_genes)
        mat = coo_matrix(
            (data_vals, (row_indices, gene_coords)), 
            shape=(len(indices), self.n_genes)
        )
        
        # Convert to dense torch tensor
        # float32 is standard
        batch_x = torch.from_numpy(mat.toarray()).float()
        
        return batch_x, torch.tensor(indices)

    def __getstate__(self):
        """
        Drop non-picklable TileDB handles when forked/spawned.
        """
        state = self.__dict__.copy()
        state["_ctx"] = None
        state["_array"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Handles will be recreated lazily

class TileDBDataset(Dataset):
    def __init__(self, tiledb_path: str, indices: np.ndarray, n_genes: int):
        super().__init__()
        self.tiledb_path = tiledb_path
        self.indices = indices
        self.n_genes = n_genes

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Only return the global index. 
        # The heavy lifting is done in the Collator.
        return self.indices[idx]
