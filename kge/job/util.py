import torch
from torch import Tensor
from typing import List, Union
from kge import Dataset

def get_coords_from_spo_batch(
    batch: Union[Tensor, List[Tensor]], dataset: Dataset, query_types: List[str], set_type: str
) -> torch.Tensor:
    if type(batch) is list:
        batch = torch.cat(batch).reshape((-1, 3)).int()
    coords = {}
    for query_type in query_types:
        if query_type == 'sp_to_o':
            sp_index = dataset.index(f"{set_type}_{query_type}")
            coords[query_type] = sp_index.get_all(batch[:, [0, 1]])
        if query_type == 'po_to_s':
            po_index = dataset.index(f"{set_type}_{query_type}")
            coords[query_type] = po_index.get_all(batch[:, [1, 2]])
        if query_type == 'so_to_p':
            so_index = dataset.index(f"{set_type}_{query_type}")
            coords[query_type] = so_index.get_all(batch[:, [0, 2]])
    num_elements = 0
    for query_type in coords.keys():
        coords[query_type][:, 1] += num_elements
        num_elements += dataset.num_relations() if query_type is 'so_to_p' else dataset.num_entities()
    coords = torch.cat(
        tuple(coords.values())
    )
    return coords

def get_sp_po_coords_from_spo_batch(
    batch: Union[Tensor, List[Tensor]], num_entities: int, sp_index: dict, po_index: dict
) -> torch.Tensor:
    """Given a set of triples , lookup matches for (s,p,?) and (?,p,o).

    Each row in batch holds an (s,p,o) triple. Returns the non-zero coordinates
    of a 2-way binary tensor with one row per triple and 2*num_entites columns.
    The first half of the columns correspond to hits for (s,p,?); the second
    half for (?,p,o).

    """
    if type(batch) is list:
        batch = torch.cat(batch).reshape((-1, 3)).int()
    sp_coords = sp_index.get_all(batch[:, [0, 1]])
    po_coords = po_index.get_all(batch[:, [1, 2]])
    po_coords[:, 1] += num_entities
    coords = torch.cat(
        (
            sp_coords,
            po_coords
        )
    )

    return coords


def coord_to_sparse_tensor(
    nrows: int, ncols: int, coords: Tensor, device: str, value=1.0, row_slice=None
):
    if row_slice is not None:
        if row_slice.step is not None:
            # just to be sure
            raise ValueError()

        coords = coords[
            (coords[:, 0] >= row_slice.start) & (coords[:, 0] < row_slice.stop), :
        ]
        coords[:, 0] -= row_slice.start
        nrows = row_slice.stop - row_slice.start

    if device == "cpu":
        labels = torch.sparse.FloatTensor(
            coords.long().t(),
            torch.ones([len(coords)], dtype=torch.float, device=device) * value,
            torch.Size([nrows, ncols]),
        )
    else:
        labels = torch.cuda.sparse.FloatTensor(
            coords.long().t(),
            torch.ones([len(coords)], dtype=torch.float, device=device) * value,
            torch.Size([nrows, ncols]),
            device=device,
        )

    return labels
