from torch.utils.data.dataset import Dataset


class TripleDataset(Dataset):
    """Dataset wrapping data, target, and weight tensors.

    Each sample will be retrieved by indexing all tensors along the first
    dimension.

    Arguments:
        data_tensor (Tensor): contains sample data.
        target_tensor (Tensor): contains sample targets (labels).
        sample_weights_tensor (Tensor): contains sample weights.
    """

    def __init__(self, data_tensor, target_tensor, sample_weights_tensor):
        assert data_tensor.size(0) == target_tensor.size(0)
        assert sample_weights_tensor.size(0) == target_tensor.size(0)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.sample_weights_tensor = sample_weights_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index],\
            self.sample_weights_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)
