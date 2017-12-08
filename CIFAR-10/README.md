# Distributed-TensorFlow-Example
About a cluster of TensorFlow servers, and how to distribute a computation graph across that cluster

# Requirements
- Python 3.5.2
- TensorFlow >= 1.4.0 (tf.data.FixedLengthRecordDataset)
- horovod

## How to Run 
```
python cifar10_download_and_extract.py

srun -n 4 --mpi=pmi2 --partition=k80 --gres=gpu:4 python cifar10_main.py --data_dir=data/cifar10_data
```

## Reference
[horovod/examples/tensorflow_mnist.py](https://github.com/uber/horovod/blob/master/examples/tensorflow_mnist.py)

[models/official/resnet/cifar10_main.py](https://github.com/tensorflow/models/blob/master/official/resnet/cifar10_main.py)


