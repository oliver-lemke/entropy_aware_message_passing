from abc import abstractmethod

from torch_geometric.data import Dataset


class BaseDataset(Dataset):
    @abstractmethod
    def __init__(self):
        super().__init__()
        raise NotImplementedError()

    @abstractmethod
    def get_input_output_dim(self) -> (int, int):
        raise NotImplementedError()
