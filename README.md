# tfcluster

this is a distributed training to support tensorflow and caffe, below is some features:
1. based on mesos's container using docker image, but not docker
2. support both cpu and gpu 
3. Data Parallelism and Model Parallelism for tensorflow
4. only Data Parallelism for caffe
5.support hdfs, you can put your datas, including data, model and so on

# requirement
mesos, distributed file system(hdfs, mfs), docker,  [caffe_hdfs](https://github.com/qingzew/caffe_hdfs)
