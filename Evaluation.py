
import weka.core.jvm as jvm
jvm.start(max_heap_size="12g")
data_dir = "---------------"
from weka.core.converters import Loader
loader = Loader(classname="weka.core.converters.ArffLoader")
data = loader.load_file(data_dir + "train_f1_4_load.txt") 
from weka.clusterers import Clusterer
clusterer = Clusterer(classname="weka.clusterers.SimpleKMeans", options=["-N", “500”])
clusterer.build_clusterer(data)
