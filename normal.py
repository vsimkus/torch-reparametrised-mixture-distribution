import numpy as np
import torch
from torch.distributions.normal import Normal


class StableNormal(Normal):
    """
    Add stable cdf for implicit reparametrization, and stable _log_cdf.
    """
    
    # Override default
    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return ndtr(self._standardise(value))
    
    # NOTE: This is not necessary for implicit reparam.
    def _log_cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return log_ndtr(self._standardise(value))
    
    def _standardise(self, x):
        return (x - self.loc) * self.scale.reciprocal()

#
# Below are based on the investigation in https://github.com/pytorch/pytorch/issues/52973#issuecomment-787587188
# and implementations in SciPy and Tensorflow Probability
#

def ndtr(value: torch.Tensor):
    """
    Standard Gaussian cumulative distribution function.
    Based on the SciPy implementation of ndtr
    https://github.com/scipy/scipy/blob/master/scipy/special/cephes/ndtr.c#L201-L224
    """
    sqrt_half = torch.sqrt(torch.tensor(0.5, dtype=value.dtype))
    x = value * sqrt_half
    z = torch.abs(x)
    y = 0.5 * torch.erfc(z)
    output = torch.where(z < sqrt_half,
                        0.5 + 0.5 * torch.erf(x),
                        torch.where(x > 0, 1 - y, y))
    return output


# log_ndtr uses different functions over the ranges
# (-infty, lower](lower, upper](upper, infty)
# Lower bound values were chosen by examining where the support of ndtr
# appears to be zero, relative to scipy's (which is always 64bit). They were
# then made more conservative just to be safe. (Conservative means use the
# expansion more than we probably need to.)
LOGNDTR_FLOAT64_LOWER = -20.
LOGNDTR_FLOAT32_LOWER = -10.

# Upper bound values were chosen by examining for which values of 'x'
# Log[cdf(x)] is 0, after which point we need to use the approximation
# Log[cdf(x)] = Log[1 - cdf(-x)] approx -cdf(-x). We chose a value slightly
# conservative, meaning we use the approximation earlier than needed.
LOGNDTR_FLOAT64_UPPER = 8.
LOGNDTR_FLOAT32_UPPER = 5.

def log_ndtr(value: torch.Tensor):
    """
    Standard Gaussian log-cumulative distribution function.
    This is based on the TFP and SciPy implementations.
    https://github.com/tensorflow/probability/blame/master/tensorflow_probability/python/internal/special_math.py#L156-L245
    https://github.com/scipy/scipy/blob/master/scipy/special/cephes/ndtr.c#L316-L345
    """
    dtype = value.dtype
    if dtype == torch.float64:
        lower, upper = LOGNDTR_FLOAT64_LOWER, LOGNDTR_FLOAT64_UPPER
    elif dtype == torch.float32:
        lower, upper = LOGNDTR_FLOAT32_LOWER, LOGNDTR_FLOAT32_UPPER
    else:
        raise TypeError(f'dtype={value.dtype} is not supported.')
    
    # When x < lower, then we perform a fixed series expansion (asymptotic)
    # = log(cdf(x)) = log(1 - cdf(-x)) = log(1 / 2 * erfc(-x / sqrt(2)))
    # = log(-1 / sqrt(2 * pi) * exp(-x ** 2 / 2) / x * (1 + sum))
    # When x >= lower and x <= upper, then we simply perform log(cdf(x))
    # When x > upper, then we use the approximation log(cdf(x)) = log(1 - cdf(-x)) \approx -cdf(-x)
    # The above approximation comes from Taylor expansion of log(1 - y) = -y - y^2/2 - y^3/3 - y^4/4 ...
    # So for a small y the polynomial terms are even smaller and negligible.
    # And we know that for x > upper, y = cdf(x) will be very small.
    return torch.where(value > upper,
                       -ndtr(-value),
                       torch.where(value >= lower,
                                   torch.log(ndtr(value)),
                                   log_ndtr_series(value)))

def log_ndtr_series(value: torch.Tensor, num_terms=3):
    """
    Function to compute the asymptotic series expansion of the log of normal CDF
    at value.
    This is based on the SciPy implementation.
    https://github.com/scipy/scipy/blob/master/scipy/special/cephes/ndtr.c#L316-L345
    """
    # sum = sum_{n=1}^{num_terms} (-1)^{n} (2n - 1)!! / x^{2n}))
    value_sq = value ** 2
    t1 = -0.5 * (np.log(2 * np.pi) + value_sq) - torch.log(-value)
    t2 = torch.zeros_like(value)
    value_even_power = value_sq.clone()
    double_fac = 1
    multiplier = -1
    for n in range(1, num_terms + 1):
        t2.add_(multiplier * double_fac / value_even_power)
        value_even_power.mul_(value_sq)
        double_fac *= (2 * n - 1)
        multiplier *= -1
    return t1 + torch.log1p(t2)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import scipy.special as ss
    
    x = torch.linspace(-30, 10, 40000, dtype=torch.float32)
    out = log_ndtr(x)
    plt.plot(x.numpy(), abs(out.numpy() - ss.log_ndtr(x.numpy())), label='abs(PyTorch - SciPy) (float32)')
    plt.legend()
    plt.show()

    x = torch.linspace(-30, 10, 40000, dtype=torch.float64)
    out = log_ndtr(x)
    plt.plot(x.numpy(), abs(out.numpy() - ss.log_ndtr(x.numpy())), label='abs(PyTorch - SciPy) (float64)')
    plt.legend()
    plt.show()

    x = torch.linspace(-30, 10, 40000, dtype=torch.float32)
    plt.plot(x.numpy(), abs(torch.distributions.Normal(0, 1).cdf(x).numpy() - ss.ndtr(x.numpy())), label='Old')
    plt.plot(x.numpy(), abs(ndtr(x).numpy() - ss.ndtr(x.numpy())), label='New')
    plt.title('Float32')
    plt.legend()
    plt.show()

    x = torch.linspace(-30, 10, 40000, dtype=torch.float64)
    plt.plot(x.numpy(), abs(torch.distributions.Normal(0, 1).cdf(x).numpy() - ss.ndtr(x.numpy())), label='Old')
    plt.plot(x.numpy(), abs(ndtr(x).numpy() - ss.ndtr(x.numpy())), label='New')
    plt.title('Float64')
    plt.legend()
    plt.show()