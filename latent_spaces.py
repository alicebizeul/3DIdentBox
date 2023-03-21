"""Classes that combine spaces with specific probability densities."""

from typing import Callable, List
from spaces import Space
import torch


class LatentSpace:
    """Combines a topological space with a marginal and conditional density to sample from."""

    def __init__(
        self, space: Space, sample_marginal: Callable, sample_conditional: Callable
    ):
        self.space = space
        self._sample_marginal = sample_marginal
        self._sample_conditional = sample_conditional

    @property
    def sample_conditional(self):
        if self._sample_conditional is None:
            raise RuntimeError("sample_conditional was not set")
        return lambda *args, **kwargs: self._sample_conditional(
            self.space, *args, **kwargs
        )

    @sample_conditional.setter
    def sample_conditional(self, value: Callable):
        assert callable(value)
        self._sample_conditional = value

    @property
    def sample_marginal(self):
        if self._sample_marginal is None:
            raise RuntimeError("sample_marginal was not set")
        return lambda *args, **kwargs: self._sample_marginal(
            self.space, *args, **kwargs
        )

    @sample_marginal.setter
    def sample_marginal(self, value: Callable):
        assert callable(value)
        self._sample_marginal = value

    @property
    def dim(self):
        return self.space.dim


class ProductLatentSpace(LatentSpace):
    """A latent space which is the cartesian product of other latent spaces."""

    def __init__(self, spaces: List[LatentSpace]):
        self.spaces = spaces

    def sample_conditional(self, means, params, size, **kwargs):
        x = []
        for i, s in enumerate(self.spaces):
            if len(means.shape) == 1:
                z_s = means[i]
            else:
                z_s = means[:, i]
            x.append(s.sample_conditional(mean=z_s, params=params[i], size=size, **kwargs))
        return torch.cat(x, -1)

    def sample_marginal(self, means, params, size, **kwargs):
        x = [s.sample_marginal(means[:,i], params[i], size=size, **kwargs) for i, s in enumerate(self.spaces)]
        return torch.cat(x, -1)

    def sample_marginal_causal(self, std, size, first_content, **kwargs):
        x = [s.sample_marginal(torch.as_tensor([0.0]),torch.as_tensor([0.0]), size=size, **kwargs) for i, s in enumerate(self.spaces)]
        final_x = []
        for i, s in enumerate(self.spaces):
            if i==1 and std[i] is not None:
                if first_content:
                    final_x.append(s.sample_marginal(x[-4],std[i]))
                else:
                    final_x.append(s.sample_marginal(x[-2],std[i]))
            elif i==6 and std[i] is not None:
                if first_content:
                    final_x.append(s.sample_marginal(x[1],std[i]))
                else:
                    final_x.append(s.sample_marginal(x[-2],std[i]))
            elif i==8 and std[i] is not None:
                if first_content:
                    final_x.append(s.sample_marginal(x[-4],std[i]))
                else:
                    final_x.append(s.sample_marginal(x[1],std[i]))
            elif i in (0,2,3,4,5,7,9): final_x.append(x[i])

        final_final_x = []
        for i, s in enumerate(self.spaces):

            if i==0 and std[i] is not None: final_final_x.append(s.sample_marginal(x[1],std[i]))
            elif i==5 and std[i] is not None:final_final_x.append(s.sample_marginal(x[-4],std[i]))
            elif i==7 and std[i] is not None:final_final_x.append(s.sample_marginal(x[-2],std[i]))
            elif i in (1,2,3,4,6,8,9): final_x.append(x[i])

        return torch.cat(final_final_x, -1)

    @property
    def dim(self):
        return sum([s.dim for s in self.spaces])
