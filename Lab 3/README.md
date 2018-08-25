# Assignment 3

The working code (assuming the data set is placed in the right folder) is in `Assignment3.py`, `helpers.py`, `analyse_hyperparam.py` and `methods.py`.

## Summary of files
* `methods.py` - contains the code used for training the two-layer network.
* `helpers.py` - methods that help to read the data from the CIFAR-10 dataset.
* `analyse_hyperparam.py` - contains code to analyse and find the best performing hyper-parameters (given files with results).
* `Assignment3.py` - mix of methods that were used during the assignment (for example, to search for hyper parameters, testing gradients, etc).

## Notes when using batchnorm version
Although it is possible to simply run the fit method and then test it on either the test or validation sets (which will use the exponential average obtained from training), I have noticed that much better results were obtained (factor 0.2-0.3 better) when computing the `mean` and `variance` of the training set directly and use it on the test and validation sets

```
# acc = exp average
# acc2 = actual mean and var of the training set

clf.fit(X_train, Y_train, X_test,X_test)
acc = clf.compute_accuracy(X_val, Y_val)

p, cache = clf.eval_classifier(X_train)

clf.mu = cache[3]
clf.v = cache[4]

acc2 = clf.compute_accuracy(X_val, Y_val)
```
