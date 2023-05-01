# Autoencoders using Uncertainty Prediction

Train VAE <br>
``python3 main.py``

Train UPAE <br>
``python3 main.py --u``
- not finished on UPAE, working on getting reconstruction score with noise variance 


## cara-main.ipynb (current progress)


### models3.py
- <b>Sampling()</b>  this gets sample from the latent space. gets mean and logvar 
from latent space then gets random vector(epsilon). This is to have randomness in the output 
and also generate similar data points from training images. <br>
- The mean represents the expected value of the distribution, which can be thought of as the most likely value for the latent variable given the input. 
- The log variance represents the logarithm of the variance of the distribution, which measures how much the distribution varies from its mean.
<br>
- [reference for AE](https://pyimagesearch.com/2020/02/17/autoencoders-with-keras-tensorflow-and-deep-learning/)
- [Keras AE reference](https://blog.keras.io/building-autoencoders-in-keras.html)


### main.py



