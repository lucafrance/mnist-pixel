# Distinguishing pairs of classes in MNIST and Fashion-MNIST with just one pixel 
Most pairs of classes in MNIST as well Fashion-MNIST can be distinguished with just one pixels. This code calculates by how much and visualizes the results.
The results are described in [this article](https://lucafrance.github.io/2021/04/05/mnist-pairwise-one-pixel.html).

### Usage
To download the datasets to `data`.
```
python download_data.py
```
To run the logistic regressions and save the results to csv-files in `csv` and `results.txt`. 
```
python evaluate_models.py
```
To visualize the results and save them to `images`.
```
python visualizations.py
```
To perform all the above operations in PowerShell.
```
.\run_all.ps1
```

### References
* [MNIST](http://yann.lecun.com/exdb/mnist/)
* [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)
* [Variance explained - Exploring handwritten digit classification: a tidy analysis of the MNIST dataset](http://varianceexplained.org/r/digit-eda/)
* [Zalando Research](https://research.zalando.com/)
