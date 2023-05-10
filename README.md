# Autoencoders using Uncertainty Prediction

Train VAE <br>
``python3 main.py``

Train UPAE <br>
``python3 main.py --u``



## cara-main.ipynb (current progress)

- :heavy_check_mark:  UPAE and Vanilla training loop <br>
- :heavy_check_mark:  validation with performance metrics (learning curve and validation loss) <br> 
    to interpret curves use [this](https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/)
    <br>
- :heavy_check_mark: testing - show reconstructed image <br>
- :o:	Testing with performance metrics (AUC, F1, EER) <br>	
- :o: save each reconstruction in training loop to make gif on training progress <br>
- :o: in testing, get image highlighting : (1) variance -mse (2) pixel-wise uncertainty (3) area og abnormality  <br>
- :o: in testing, show difference between reconstruction image output of normal and abnormal <br>
- :o: implement callbacks to use tensorboard. basing on [this](https://keras.io/guides/writing_your_own_callbacks/)
- :o: use callback to automatically save model / best model. [reference](https://keras.io/api/callbacks/model_checkpoint/#:~:text=Callback%20to%20save%20the%20Keras,training%20from%20the%20state%20saved.)


### Autoencoder Structure
encoder:
Latent_enc: did not seperate anymore
Latent_dec:
decoder:

### Vanilla AE
- computed for MSE between input and reconstruction. 
- 

### UPAE
- got mean, and logvariance after decoder. not in encoder. 
    - logvar - differenc
- 


### references
<br>

[reference for AE](https://pyimagesearch.com/2020/02/17/autoencoders-with-keras-tensorflow-and-deep-learning/)

[Keras AE reference](https://blog.keras.io/building-autoencoders-in-keras.html)





