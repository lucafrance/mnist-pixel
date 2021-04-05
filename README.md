mnist_reader viene da qui, da citare nela licenza MIT https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py

http://varianceexplained.org/r/digit-eda/

* claim that you can predict pairs of mnist digits with one pixels
  * makes sense looking at the heatmap
  * if we do it precisely, we see...
    * logistic regression
      * for each pair of digits
      * the pixel that predicts best
    * accuracy of prediction for each pair of digit
* what if we go all in and try to predict all digits with one pixels?
* how does it compare to full logistic regression?
* how does fashion-mnist compare?
