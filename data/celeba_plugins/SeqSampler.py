from torch.utils.data import Sampler
from torch import tensor

class SeqSampler(Sampler):
    """This class implements a subclass of Sampler for sampling according to
    given indices.

    Attributes:
        indices (list): A sequence of indices which is to be sampled.
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        # Option for just using the indices
        print(self.indices[0:100])
        return iter((self.indices[0:100]))

    def __len__(self):
        return len(self.indices)
