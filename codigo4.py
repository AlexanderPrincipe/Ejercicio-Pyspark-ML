from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.feature import VectorIndexer, ChiSqSelector		
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def tipo_posiciones(p):
    # Ofensivo
    if p == 'RF' or p == 'ST' or p == 'LF' or p == 'RS' or p == 'LS' or p == 'CF' or p == 'LW' or p == 'RCM' or p == 'LCM' or p == 'LDM' or p == 'CAM' or p == 'CDM' or p == 'RM' or p == 'LAM' or p == 'LM' or p == 'RDM' or p == 'RW' or p == 'CM' or p == 'RAM':
        return 1 
    # Defensivo
    elif p == 'RCB' or p == 'CB' or p == 'LCB' or p == 'LB' or p == 'RB' or p == 'RWB' or p == 'LWB' or p == 'GK':
        return 0
    else:
        return 1

def leer_df():
	conf = SparkConf().setAppName("Tarea4").setMaster("local")
	sc = SparkContext(conf=conf)

	sqlContext = SQLContext(sc)

	# Leemos el CSV
	rdd = sqlContext.read.csv("data_real.csv", header=True).rdd

	rdd = rdd.map(
        lambda x: (int(x[54]),int(x[55]),int(x[56]),int(x[57]),int(x[58]), int(x[59]),
		                  int(x[60]), int(x[61]), int(x[62]),int(x[63]), int(x[64]), int(x[65]) , int(x[66]),
		                  int(x[67]), int(x[68]), int(x[69]),int(x[70]), int(x[71]), int(x[72]) , int(x[73]),
		                  int(x[74]), int(x[75]), int(x[76]),int(x[77]), int(x[78]), int(x[79]) , int(x[80]),
		                  int(x[81]), int(x[82]), int(x[83]),int(x[84]), int(x[85]), int(x[86]) , int(x[87]), tipo_posiciones((x[21]))))
	df = rdd.toDF(["Crossing","Finishing","HeadingAccuracy","ShortPassing","Volleys","Dribbling","Curve",
                   "FKAccuracy", "LongPassing","BallControl","Acceleration","SprintSpeed","Agility","Reactions",
                   "Balance","ShotPower","Jumping","Stamina","Strength","LongShots","Aggression",
                   "Interceptions","Positioning","Vision","Penalties","Composure","Marking","StandingTackle",
                   "SlidingTackle","GKDiving","GKHandling","GKKicking","GKPositioning","GKReflexes", "Position"])
	return df



def leer_df_categoricos():
	conf = SparkConf().setAppName("Tarea4").setMaster("local")
	sc = SparkContext(conf=conf)

	sqlContext = SQLContext(sc)

	# Leemos el CSV
	rdd = sqlContext.read.csv("data_real.csv", header=True).rdd

	rdd = rdd.map(
        lambda x: (int(x[54]),int(x[55]),int(x[56]),int(x[57]),int(x[58]), int(x[59]),
		                  int(x[60]), int(x[61]), int(x[62]),int(x[63]), int(x[64]), int(x[65]) , int(x[66]),
		                  int(x[67]), int(x[68]), int(x[69]),int(x[70]), int(x[71]), int(x[72]) , int(x[73]),
		                  int(x[74]), int(x[75]), int(x[76]),int(x[77]), int(x[78]), int(x[79]) , int(x[80]),
		                  int(x[81]), int(x[82]), int(x[83]),int(x[84]), int(x[85]), int(x[86]) , int(x[87]), tipo_posiciones((x[21]))))
	df = rdd.toDF(["Crossing","Finishing","HeadingAccuracy","ShortPassing","Volleys","Dribbling","Curve",
                   "FKAccuracy", "LongPassing","BallControl","Acceleration","SprintSpeed","Agility","Reactions",
                   "ShotPower","Jumping","Stamina","Strength","LongShots",
                   "Interceptions","Positioning","Vision","Marking","StandingTackle",
                   "Position"])

def feature_selection(df):
	assembler = VectorAssembler(
		inputCols=["Crossing","Finishing","HeadingAccuracy","ShortPassing","Volleys","Dribbling","Curve",
                   "FKAccuracy", "LongPassing","BallControl","Acceleration","SprintSpeed","Agility","Reactions",
                   "Balance","ShotPower","Jumping","Stamina","Strength","LongShots","Aggression",
                   "Interceptions","Positioning","Vision","Penalties","Composure","Marking","StandingTackle",
                   "SlidingTackle","GKDiving","GKHandling","GKKicking","GKPositioning","GKReflexes"],
		outputCol="features")
	df = assembler.transform(df)

	indexer = VectorIndexer(
		inputCol="features", 
		outputCol="indexedFeatures",
		maxCategories=4)
	
	df = indexer.fit(df).transform(df)

	# Seleccionamos features que mas suman al modelo
	selector = ChiSqSelector(
		numTopFeatures=2,
		featuresCol="indexedFeatures",
		labelCol="Position",
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

	# Entrenamiento
	entrenador = DecisionTreeClassifier(
		labelCol="Position", 
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
