import os
from urllib.request import urlretrieve


if __name__ == "__main__":
    
    for path in [os.path.join("data", "mnist"), os.path.join("data", "fashion")]:
        os.makedirs(path, exist_ok=True)
    
    print("Downloading MNIST...")
    mnist_urls = ["http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
                  "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
                  "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
                  "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",]
    for url in mnist_urls:
        filename = url.split("/")[-1]
        urlretrieve(url, filename=os.path.join("data", "mnist", filename))
        
    print("Downloading Fashion-MNIST...")
    fashion_urls = ["http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
                    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
                    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
                    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",]
    for url in fashion_urls:
        filename = url.split("/")[-1]
        urlretrieve(url, filename=os.path.join("data", "fashion", filename))
