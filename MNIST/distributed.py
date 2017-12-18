#/usr/bin/python
#-*-coding=utf-8-*-

# ps 节点执行： 
# CUDA_VISIBLE_DEVICES='' python distribute.py --ps_hosts=192.168.1.100:2222 --worker_hosts=192.168.1.100:2224,192.168.1.100:2225 --job_name=ps --task_index=0

# worker 节点执行:
# CUDA_VISIBLE_DEVICES=0  python distribute.py --ps_hosts=192.168.1.100:2222 --worker_hosts=192.168.1.100:2224,192.168.1.100:2225 --job_name=worker --task_index=0

# CUDA_VISIBLE_DEVICES='' python distribute.py --ps_hosts=192.168.1.100:2222 --worker_hosts=192.168.1.100:2224,192.168.1.100:2225 --job_name=worker --task_index=1 

# CUDA_VISIBLE_DEVICES=0  USE GPU0
# CUDA_VISIBLE_DEVICES='' USE CPU

import numpy as np 
import tensorflow as tf 

# Define parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('lr', 0.00003, 'Initial learning rate')
tf.app.flags.DEFINE_integer('steps_to_validate', 1000, 'Step to validate and print loss')

# For distributed
tf.app.flags.DEFINE_string("ps_hosts", "", "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "", "Comma-seperated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("issync", 0, "Whether to adopt Distributed Synchronization Mode, 1: sync, 0:async")

# Hyperparameters
learning_rate = FLAGS.lr
steps_to_validate = FLAGS.steps_to_validate

def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker":worker_hosts})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    issync = FLAGS.issync
    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        with tf.device(tf.train.replica_device_setter(
            worker_device = "/job:worker/task:%d" % FLAGS.task_index,
            cluster=cluster
        )):
            global_step = tf.Variable(0, name='global_step', trainable=False)

            input = tf.placeholder("float")
            label = tf.placeholder("float")

            weight = tf.get_variable("weight", [1], tf.float32, initializer=tf.random_normal_initializer())
            biase = tf.get_variable("biase", [1], tf.float32, initializer=tf.random_normal_initializer())
            pred = tf.multiply(input, weight) + biase

            loss_value = loss(label, pred)
            loss_value = tf.reshape(loss_value,[])
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)

            grads_and_vars = optimizer.compute_gradients(loss_value)

            if issync == 1:
                # Update gradients in Synchronization Mode
                rep_op = tf.train.SyncReplicasOptimizer(optimizer, 
                                                        replicas_to_aggregate=len(worker_hosts),
                                                        replica_id=FLAGS.task_index,
                                                        total_num_replicas=len(worker_hosts),
                                                        use_locking=True
                                                        )

                train_op = rep_op.apply_gradients(grads_and_vars,
                                                  global_step=global_step
                                                 )

                init_token_op = rep_op.get_init_tokens_op()
                chief_queue_runner = rep_op.get_chief_queue_runner()

            else:
                # Update gradients in Asynchronization Mode
                train_op = optimizer.apply_gradients(grads_and_vars,
                                                     global_step=global_step
                                                    )

                saver = tf.train.Saver()
                tf.summary.scalar('cost', loss_value)
                summary_op = tf.summary.merge_all()

                init_op = tf.global_variables_initializer()

            sv = tf.train.Supervisor(is_chief=(FLAGS.task_index==0),
                                     logdir="./checkpoint/",
                                     init_op=init_op,
                                     summary_op=None, #summary_op,
                                     saver=saver,
                                     global_step=global_step,
                                     save_model_secs=60                                        
                                    )                

            with sv.prepare_or_wait_for_session(server.target) as sess:
                # If is Synchronization Mode
                if FLAGS.task_index == 0 and issync == 1:
                    sv.start_queue_runners(sess, [chief_queue_runner])
                    sess.run(init_token_op)
                step = 0
                while step < 1000000:
                    train_x = np.random.randn(1)
                    train_y = 2 * train_x + np.random.randn(1) * 0.33 + 10
                    _, loss_v, step, summary = sess.run([train_op, loss_value, global_step, summary_op], feed_dict={input:train_x, label:train_y})
                    if step % steps_to_validate == 0:
                        w, b = sess.run([weight, biase])
                        print("step: %d, weight: %f, biase: %f, loss: %f" %(step, w, b, loss_v))
                        sv.summary_computed(sess, summary)

            sv.stop()

def loss(label, pred):
    return tf.square(label - pred)

if __name__ == "__main__":
    tf.app.run()          



