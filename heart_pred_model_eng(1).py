from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.feature import VectorIndexer, ChiSqSelector		
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def leer_df():
	conf = SparkConf().setAppName("HeartPred").setMaster("local")
	sc = SparkContext(conf=conf)

	sqlContext = SQLContext(sc)

	# Leemos el CSV
	rdd = sqlContext.read.csv("heart.csv", header=True).rdd

	rdd = rdd.map(
		lambda x: ( int(x[0]), int(x[1]), int(x[2]), 
			int(x[3]), int(x[4]), 
			int(x[5]) , int(x[6]), int(x[7]), int(x[8]), float(x[9]),
			int(x[10]), int(x[11]), int(x[12]) , int(x[13]) ))
	df = rdd.toDF(["age","sex","cp","trestbps","chol", "fbs", 
		"restecg","thalach","exang","oldpeak","slope","ca","thal","target"])

	return df

def leer_df_categoricos():
	conf = SparkConf().setAppName("HeartPred").setMaster("local")
	sc = SparkContext(conf=conf)

	sqlContext = SQLContext(sc)

	# Leemos el CSV
	rdd = sqlContext.read.csv("heart_2.csv", header=True).rdd

	rdd = rdd.map(
		lambda x: ( int(x[0]), int(x[1]), 
			int(x[2]) , int(x[3]), int(x[4]),
			int(x[5]), int(x[6]), int(x[7]) , int(x[8]) ))
	df = rdd.toDF(["sex","cp", "fbs",
		"restecg","exang","slope","ca","thal","target"])

	return df

def feature_selection(df):
	assembler = VectorAssembler(
		inputCols=["age","sex","cp","trestbps","chol", "fbs", "restecg","thalach","exang",
					"oldpeak","slope","ca","thal"],
		outputCol="features")
	df = assembler.transform(df)

	indexer = VectorIndexer(
		inputCol="features", 
		outputCol="indexedFeatures",
		maxCategories=4)
	
	df = indexer.fit(df).transform(df)

	# Seleccionamos features que mas suman al modelo
	selector = ChiSqSelector(
		numTopFeatures=4,
		featuresCol="indexedFeatures",
		labelCol="target",
		outputCol="selectedFeatures")
	resultado = selector.fit(df).transform(df)
	resultado.select("features", "selectedFeatures").show()

def entrenamiento(df):
	# Vectorizo
	df = df.select("cp", "restecg", "thal", "target")
	assembler = VectorAssembler(
		inputCols=["cp","restecg","thal"],
		outputCol="features")
	df = assembler.transform(df)

	# Dividir nuestro dataset
	(training_df, test_df) = df.randomSplit([0.7, 0.3])

	## ENTRENAMIENTO
	entrenador = DecisionTreeClassifier(
		labelCol="target", 
		featuresCol="features")
	pipeline = Pipeline(stages=[entrenador])
	model = pipeline.fit(training_df)

	predictions_df = model.transform(test_df)

	evaluator = MulticlassClassificationEvaluator(
		labelCol="target",
		predictionCol="prediction",
		metricName="accuracy")

	exactitud = evaluator.evaluate(predictions_df)
	print("Exactitud: {}".format(exactitud))


def main():
	df = leer_df()
	feature_selection(df)
	#entrenamiento(df)
	


if __name__ == "__main__":
	main()
















