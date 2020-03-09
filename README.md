### Explaining and Harnessing Adversarial Examples(FGSM) - ICLR 2015
 
This is the implementation in pytorch of FGSM based [Explaining and Harnessing Adversarial Examples(2015)](https://arxiv.org/abs/1412.6572)

Use Two dataset : MNIST, ImageNet(later)

quick start
<pre>
<code>
python fgsm_mnist.py
</code>
</pre>

If you want to choose Hyperparameter or Network type, see option
<pre>
<code>
python fgsm_mnist.py -h
</code>
</pre>

## Result
### MNIST DataSet
Shallow model(fc layer x 2) - epsilon: 0.25
- use default argument(hyperparameter) 
1. Test Accuracy : 99%
2. Adversarial Test Accuracy: 1%
3. Misclassification : 9696/10000

Shallow model(fc layer x 2) - epsilon: 0.25 / data_normalization: True
- If use dataset normalization, you can somewhat resist Adversarial Example.
1. Test Accuracy: 98%
2. Adversarial Test Accuracy: 89%
3. Misclassification: 938/10000

