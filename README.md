# Reparametrisable PyTorch MixtureSameFamily distribution

PyTorch implementation of the implicit reparametrisation trick for mixture distributions based on [Figurnov et al., 2019, "Implicit Reparameterization Gradients"](https://papers.nips.cc/paper/2018/hash/92c8c96e4c37100777c7190b76d28233-Abstract.html) and the implementation in [Tensorflow Probability](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/MixtureSameFamily).

Can be readily used for variational inference with mixture distribution variational families.

Remarks:

* For multivariate mixtures, the class is currently implemented when the mixture component distributions fully factorise. 
* Also added a `StableNormal` distribution, which overrides the default `cdf` method with a more stable implementation from <https://github.com/pytorch/pytorch/issues/52973#issuecomment-787587188>. The implementation also provides a `_log_cdf` method, however it is not used for the implicit reparametrisation.
