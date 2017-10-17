# Distributed-TensorFlow-Example
About a cluster of TensorFlow servers, and how to distribute a computation graph across that cluster

# Requirements
Python 3.5.2
TensorFlow >= 1.3.0

## How to run
```
parameter server (ps)
CUDA_VISIBLE_DEVICES='' python distributed.py --ps_hosts=202.121.182.216:20300 --worker_hosts=202.121.182.216:20200 --job_name=ps --task_index=0


worker server:

CUDA_VISIBLE_DEVICES='' python distributed.py --ps_hosts=202.121.182.216:20300 --worker_hosts=202.121.182.216:20200 --job_name=worker --task_index=0

CUDA_VISIBLE_DEVICES=0 python distributed.py --ps_hosts=192.168.100.42:2222 --worker_hosts=202.121.182.216:20200 --job_name=worker --task_index=1

```
## Reference
[Distributed TensorFlow](https://www.tensorflow.org/versions/master/deploy/distributed)

[Deploy - TensorFlow](https://www.tensorflow.org/versions/master/deploy/)

[How to run TensorFlow on Hadoop](https://www.tensorflow.org/versions/master/deploy/hadoop)

[thewintersun/distributeTensorflowExample](https://github.com/thewintersun/distributeTensorflowExample)

tensorflow.python.framework.errors_impl.UnknownError: Could not start gRPC server

