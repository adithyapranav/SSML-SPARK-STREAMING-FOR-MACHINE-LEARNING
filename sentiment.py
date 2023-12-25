
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col,lower, regexp_replace, udf
from pyspark.sql.types import StringType

import time
import re

import models as mod
import preprocessing as pp

sc = SparkContext.getOrCreate()			#Gateway to spark
sc.setLogLevel("OFF")
ssc = StreamingContext(sc, 1)				#Streaming Fuctionality 
spark=SparkSession(sc)					#sql Fuctionality 
data = ssc.socketTextStream("localhost", 6100)

try:
	def readMyStream(rdd):
		df=spark.read.json(rdd)		#Extracting rdd
		if(not df.rdd.isEmpty()):
			df=pp.preprocess(df)
			mod.model_training(df)

except Exception as e:
	print(e)		

try:
	data.foreachRDD(lambda rdd: readMyStream(rdd))
	
except Exception as e:
	print(e)

ssc.start()
ssc.awaitTermination()
#time.sleep(1000)
ssc.stop(stopSparkContext=False)
