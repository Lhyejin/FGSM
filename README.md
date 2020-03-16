### Explaining and Harnessing Adversarial Examples(FGSM) - ICLR 2015
 
This is the implementation in pytorch of FGSM based [Explaining and Harnessing Adversarial Examples(2015)](https://arxiv.org/abs/1412.6572)

Use Two dataset : MNIST(fc layer*2), CIFAR10(googleNet)

quick start
<pre>
<code>
python fgsm_mnist.py #or
python fgsm_cifar10.py
</code>
</pre>

Run this commend for more information or help
<pre>
<code>
#MNIST Example
python fgsm_mnist.py -h
usage: fgsm_mnist.py [-h] [--batch-size N] [--test-batch-size N] [--epochs N]
                     [--lr LR] [--gamma M] [--no-cuda] [--log-interval N]
                     [--epsilon EPSILON]
                     [--dataset-normalize DATASET_NORMALIZE]
                     [--network NETWORK] [--save-model]

PyTorch MNIST Example

optional arguments:
  -h, --help           show this help message and exit
  --batch-size N       input batch size for training (default: 64)
  --test-batch-size N  input batch size for testing (default: 1000)
  --epochs N           number of epochs to train (default: 5)
  --lr LR              learning rate (default: 0.001)
  --gamma M            Learning rate step gamma (default: 0.7)
  --no-cuda            disables CUDA training
  --log-interval N     how many batches to wait before logging training status
  --epsilon EPSILON
  --dataset-normalize  input whether normalize or not (default: False)
  --network NETWORK    input Network type (Selected: fc, conv, drop / default:
                       'fc')
  --save-model         For Saving the current Model

#CIAR-10 Example
python fgsm_cifar10.py -h
usage: fgsm_cifar10.py [-h] [--batch-size N] [--test-batch-size N]
                       [--epochs N] [--lr LR] [--gamma M] [--no-cuda]
                       [--log-interval N] [--epsilon EPSILON]
                       [--dataset-normalize] [--train-mode]
                       [--model_parameter MODEL_PARAMETER] [--save-model]

PyTorch CIFAR-10 Example

optional arguments:
  -h, --help            show this help message and exit
  --batch-size N        input batch size for training (default: 100)
  --test-batch-size N   input batch size for testing (default: 100)
  --epochs N            number of epochs to train (default: 20)
  --lr LR               learning rate (default: 0.001)
  --gamma M             Learning rate step gamma (default: 0.7)
  --no-cuda             disables CUDA training
  --log-interval N      how many batches to wait before logging training
                        status
  --epsilon EPSILON
  --dataset-normalize   input whether normalize or not (default: False)
  --train-mode          input whether training or not (default: True
  --model_parameter MODEL_PARAMETER
                        if test mode, input model parameter path
  --save-model          For Saving the current Model
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

