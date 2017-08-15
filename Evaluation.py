"""
You need to install the following Python packages:
numpy
pillow
matplotlib
pygraphviz
Once these libraries are installed, you can use pip to install the remaining Python packages:
$ pip install javabridge
$ pip install python-weka-wrapper """



import weka.core.jvm as jvm
jvm.start(max_heap_size="12g")
data_dir = "---------------"
from weka.core.converters import Loader
loader = Loader(classname="weka.core.converters.ArffLoader")
data = loader.load_file(data_dir + "train_f1_4_load.txt") 
from weka.clusterers import Clusterer
clusterer = Clusterer(classname="weka.clusterers.SimpleKMeans", options=["-N", “500”])
clusterer.build_clusterer(data)
print(clusterer)
# cluster the data
for inst in data:
    cl = clusterer.cluster_instance(inst)  # 0-based cluster index
    dist = clusterer.distribution_for_instance(inst)   # cluster membership distribution
    print("cluster=" + str(cl) + ", distribution=" + str(dist))
