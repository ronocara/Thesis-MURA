# Autoencoders using Uncertainty Prediction

Train VAE
``python3 main.py``

Train UPAE
``python3 main.py --u``
- not finished on UPAE, working on getting reconstruction score with noise variance 


## cara-main.ipynb (current progress)
- keras implementation of the model. 
- structure as copied from Mao
-  metrics in testing
    - AUC, Precision, Recal, TP, TP, FP, FN
    - perfromance metrics per epoch for training and validation are available thru tensorboard
- parameters:
    - epochs: 
    - learning rate 0.01
    - shape (64,64,3)
- reconstruction error based on MSE will be calculated when can already train on set of images<br>
![REocnstructed vs Input](https://github.com/aronnicksnts/MURA-Classification/blob/577050077e6f22bc788d528e4fbb3fcd5e0d7a0c/Images/inputVreconstructed.png)


## Autoencoder
- In models.py<br>
    - keras implementation of the VAE model<br>
    - layers based on Mao's AE Implementation:<br>
            - The encoder contains four layers (each with one 4 × 4 convolution with a stride 2)<br>
            - The decoder is connected by two fully connected layers and four<br>
              transposed convolutions<br>
            - encoder = 16-32-64-64   decoder=64-64-32-16<br><br>
- added latent representation (notthe same with UPAE code found online)
- [reference for AE](https://pyimagesearch.com/2020/02/17/autoencoders-with-keras-tensorflow-and-deep-learning/)
- [Keras AE reference](https://blog.keras.io/building-autoencoders-in-keras.html)


how will you get abnormality per each pixel






