## Explaining and Harnessing Adversarial Examples(FGSM) - ICLR 2015
 
This is the implementation in pytorch of FGSM based [Explaining and Harnessing Adversarial Examples(2015)](https://arxiv.org/abs/1412.6572)

Use Two dataset : MNIST(fc layer*2), CIFAR10(googleNet)

### quick start
<pre>
<code>
python fgsm.py
</code>
</pre>

### Example
<pre>
<code>
# Run this commend for more information or help
$python fgsm.py -h
usage: fgsm.py [-h] [--batch-size N] [--test-batch-size N] [--epochs N] [--lr LR] [--gamma M] [--no-cuda]
               [--log-interval N] [--epsilon EPSILON] [--dataset-normalize] [--network NETWORK] [--save-model]
               [--dataset DATASET]

PyTorch FGSM

optional arguments:
  -h, --help           show this help message and exit
  --batch-size N       input batch size for training (default: 64)
  --test-batch-size N  input batch size for testing (default: 1000)
  --epochs N           number of epochs to train (default: 5)
  --lr LR              learning rate (default: 0.001)
  --gamma M            Learning rate step gamma (default: 0.7)
  --no-cuda            disables CUDA training
  --log-interval N     how many batches to wait before logging training status
  --epsilon EPSILON    epsilon(perturbation) of adversarial attack
  --dataset-normalize  input whether normalize or not (default: False)
  --network NETWORK    input Network type (Selected: fc, conv, drop, googlenet / default: 'fc')
  --save-model         For Saving the current Model
  --dataset DATASET    choose dataset : mnist or cifar

# MNIST Example
$python fgsm.py

# CIAR-10 Example
$python fgsm.py --dataset cifar

</code>
</pre>

## Result
### MNIST DataSet
Shallow model(fc layer x 2) - epsilon: 0.25
- use default argument(hyperparameter) 
1. Test Accuracy : 99%
2. Adversarial Test Accuracy: 1%
3. Misclassification : 9696/10000

### CIFAR-10 DataSet
GoogleNet - epsilon: 0.25
- use default argument(epoch 20)
1. Test Accuracy : 82%
2. Adversarial Test Accuracy: 10%
3. Misclassification : 9191/10000

- epoch 1000
1. Test Accuracy : 91%
2. Adverarial Test Accuracy: 12%
3. Misclassification: 8775/10000
