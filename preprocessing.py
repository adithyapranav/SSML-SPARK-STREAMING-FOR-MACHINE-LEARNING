import re
from pyspark.sql.functions import col,lower, regexp_replace, udf, when
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from nltk.stem.snowball import SnowballStemmer
from pyspark.sql.types import StringType,ArrayType, IntegerType
from array import array
  

def preprocess(rdd):

	urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
	userPattern       = '@[^\s]+' 
	alphaPattern      = r"[^a-zA-Z\s]"
	sequencePattern   = r"(.)\1\1+"
	seqReplacePattern = r"\1\1"
	
	
	user= lambda x: re.sub(userPattern,'USER',x)
	url= lambda x: re.sub(urlPattern,'URL',x)
	alpha= lambda x: re.sub(alphaPattern,'',x)
	sequence= lambda x: re.sub(sequencePattern ,seqReplacePattern,x)
	spaces= lambda x: re.sub(r'\s\s+',' ',x, flags=re.I)
	quotes= lambda x: re.sub(r'"','',x)

	rdd = rdd.withColumn('tweet1',lower(col('Tweet'))).select('Sentiment','tweet1')	
	rdd = rdd.withColumn('tweet1',udf(user,StringType())('tweet1')).select('Sentiment','tweet1')
	rdd = rdd.withColumn('tweet1',udf(url,StringType())('tweet1')).select('Sentiment','tweet1')
	rdd = rdd.withColumn('tweet1',udf(sequence,StringType())('tweet1')).select('Sentiment','tweet1')
	rdd = rdd.withColumn('tweet1',udf(spaces,StringType())('tweet1')).select('Sentiment','tweet1')
	rdd = rdd.withColumn('tweet1',udf(quotes,StringType())('tweet1')).select('Sentiment','tweet1')
	rdd = rdd.withColumn('tweet1',udf(alpha,StringType())('tweet1')).select('Sentiment','tweet1')
	rdd = rdd.withColumn('Sentiment', when(rdd.Sentiment.endswith('4'), regexp_replace(rdd.Sentiment,'4','1')).when(rdd.Sentiment.endswith('0'), regexp_replace(rdd.Sentiment,'0','0')).cast(IntegerType()))
	
	tokenizer = Tokenizer(inputCol='tweet1', outputCol='words_token')
	rdd = tokenizer.transform(rdd).select('Sentiment','words_token')

	remover = StopWordsRemover(inputCol='words_token', outputCol='words_clean')
	rdd = remover.transform(rdd).select('Sentiment', 'words_clean')
	
	stemmer = SnowballStemmer(language='english')
	stemmer_udf = udf(lambda tokens: [stemmer.stem(token) for token in tokens], ArrayType(StringType()))
	rdd = rdd.withColumn("words_stemmed", stemmer_udf("words_clean")).select('Sentiment', 'words_stemmed')	#Convert to base form

	hashingTF = HashingTF(inputCol="words_stemmed", outputCol="rawFeatures", numFeatures=200)			#Convert the word into vector
	featurizedData = hashingTF.transform(rdd)
	idf = IDF(inputCol="rawFeatures", outputCol="features")							
	idfModel = idf.fit(featurizedData)										#Adjust the weights 
	rescaledData = idfModel.transform(featurizedData)
	extracted_features = rescaledData.select('Sentiment', 'features')
	
	return extracted_features
