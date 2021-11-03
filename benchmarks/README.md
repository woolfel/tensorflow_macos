# Alternate Installation

It appears tensorflow-metal is making good progress. As of November 3, 2021 the beta can be installed by following the steps in this article
[https://betterprogramming.pub/installing-tensorflow-on-apple-m1-with-new-metal-plugin-6d3cb9cb00ca] tensorflow-metal 

A basic test with mnist show the performance is better than using mlcompute.

# Running benchmark with tensorflow-metal

After you've installed and verified tensorflow-metal works on your M1 mackbook. Here's the steps for running mnist benchmark. You will need to install tensorflow_dataset in the virtual environment with "pip install tensorflow_dataset".

1. open a terminal window
2. go to benchmarks folder
3. run "conda activate tensorflow_ml"
4. run "python3 ./tfmetal_mnist_test.py"

