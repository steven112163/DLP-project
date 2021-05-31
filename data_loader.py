from torch.utils import data


class DataLoader(data.Dataset):
    def __init__(self):
        super(DataLoader, self).__init__()
        # TODO

    def __len__(self):
        """
        Number of data
        :return: number of data
        """
        # TODO

    def __getitem__(self, index):
        """
        Get current data
        :param index: index of training/testing data
        :return: data
        """
        # TODO
