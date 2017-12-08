# Distributed-TensorFlow-Example for MNIST
About a cluster of TensorFlow servers, and how to distribute a computation graph across that cluster

# Requirements
- Python 3.5.2
- TensorFlow >= 1.3.0
- horovod

## How to run

Make sure the all nodes/hosts (ip address and ports) are at the same Local Area Network (LAN). If the IP address is not correct, there will be error `tensorflow.python.framework.errors_impl.UnknownError: Could not start gRPC server`

```
Parameter Server (ps):

CUDA_VISIBLE_DEVICES='' python distributed.py --ps_hosts=192.168.1.203:10000 --worker_hosts=192.168.1.202:10001 --job_name=ps --task_index=0

Worker Server:

CUDA_VISIBLE_DEVICES='' python distributed.py --ps_hosts=192.168.1.203:10000 --worker_hosts=192.168.1.202:10001 --job_name=worker --task_index=0
```

## Reference
[Distributed TensorFlow](https://www.tensorflow.org/versions/master/deploy/distributed)

[Deploy - TensorFlow](https://www.tensorflow.org/versions/master/deploy/)

[How to run TensorFlow on Hadoop](https://www.tensorflow.org/versions/master/deploy/hadoop)

[thewintersun/distributeTensorflowExample](https://github.com/thewintersun/distributeTensorflowExample)


