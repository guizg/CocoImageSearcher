# Import SparkSession
from pyspark.sql import SparkSession
from gluoncv import model_zoo, data, utils
import pickle
import time
import mxnet as mx
import boto3
import numpy as np
import cv2
import io

# Constants
NET = model_zoo.get_model('yolo3_darknet53_coco', pretrained=True)
MEMORY_SPARK = '5gb'
BATCH_SIZE = 1000

# Build the SparkSession
spark = SparkSession.builder \
   .master("local") \
   .appName("gluoncv") \
   .config("spark.executor.memory", MEMORY_SPARK) \
   .getOrCreate()   



def lista_arquivos():
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('cocodataset')
    files = []    
    for k, obj in enumerate(bucket.objects.all()):       
        if obj.key[-4:] == '.jpg':
            files.append(obj.key)
    return files

def process_image(path):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('cocodataset')
    # path = path.encode("ascii")
    # print(path)
    with io.BytesIO() as f:
        bucket.download_fileobj(path, f)
        f.seek(io.SEEK_SET)
        x = cv2.imdecode(np.frombuffer(f.getvalue(), dtype=np.uint8), flags=3)
        x, img = data.transforms.presets.yolo.transform_test(mx.nd.array(x), short=512)
        class_IDs, scores, _ = NET(x)
    return [class_IDs.asnumpy(), scores.asnumpy(), path]



init = 0
end = BATCH_SIZE
over = False

while(!over):
	if (end > len(files)):
		end = len(files)
		over = True
	rdd = sc.parallelize(files[init:end]).map(process_image)
	results = rdd.collect()
	pickle_out = open(name, "wb")
	pickle.dump(results, pickle_out)
	init += BATCH_SIZE
	end += BATCH_SIZE
	iter += 1	

spark.stop()
