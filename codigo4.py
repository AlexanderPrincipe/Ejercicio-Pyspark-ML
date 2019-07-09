from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, Normalizer
from pyspark.ml.feature import VectorIndexer, ChiSqSelector		
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Convertir a entero jugadores ofensivos y jugadores defensivos
def tipo_posiciones(p):
    # Ofensivo
    if p == 'RF' or p == 'ST' or p == 'LF' or p == 'RS' or p == 'LS' or p == 'CF' or p == 'LW' or p == 'RCM' or p == 'LCM' or p == 'LDM' or p == 'CAM' or p == 'CDM' or p == 'RM' or p == 'LAM' or p == 'LM' or p == 'RDM' or p == 'RW' or p == 'CM' or p == 'RAM':
        return 1 
    # Defensivo
    elif p == 'RCB' or p == 'CB' or p == 'LCB' or p == 'LB' or p == 'RB' or p == 'RWB' or p == 'LWB' or p == 'GK':
        return 0
    else:
        return 1

# Cargar el CSV
def leer_df():
	conf = SparkConf().setAppName("Tarea4").setMaster("local")
	sc = SparkContext(conf=conf)

	sqlContext = SQLContext(sc)
	
	# Creacion de rdd
	rdd = sqlContext.read.csv("data.csv", header=True).rdd

	# Filtrando datos vacios
	rdd = rdd.filter(
		lambda x: (		   x[21] != None and x[21] != '' and x[54] != None and x[55] != None and x[56] != None and \
						   x[57] != None and x[58] != None and x[59] != None and x[60] != None and x[61] != None and x[62] != None and x[63] != None and x[64] != None and x[65] != None and x[66] != None and \
						   x[67] != None and x[68] != None and x[69] != None and x[70] != None and x[71] != None and x[72] != None and x[73] != None and x[74] != None and x[75] != None and x[76] != None and \
						   x[77] != None and x[78] != None and x[79] != None and x[80] != None and x[81] != None and x[82] != None and x[83] != None and x[84] != None and x[85] != None and x[86] != None and x[87] != None ))

	# Consideramos los principales features
	rdd = rdd.map(
        lambda x: (		  tipo_posiciones((x[21])) ,int(x[54].split('+')[0]),int(x[55].split('+')[0]),int(x[56].split('+')[0]),int(x[57].split('+')[0]),int(x[58].split('+')[0]), int(x[59].split('+')[0]),
		                  int(x[60].split('+')[0]), int(x[61].split('+')[0]), int(x[62].split('+')[0]),int(x[63].split('+')[0]), int(x[64].split('+')[0]), int(x[65].split('+')[0]) , int(x[66].split('+')[0]),
		                  int(x[67].split('+')[0]), int(x[68].split('+')[0]), int(x[69].split('+')[0]),int(x[70].split('+')[0]), int(x[71].split('+')[0]), int(x[72].split('+')[0]) , int(x[73].split('+')[0]),
		                  int(x[74].split('+')[0]), int(x[75].split('+')[0]), int(x[76].split('+')[0]),int(x[77].split('+')[0]), int(x[78].split('+')[0]), int(x[79].split('+')[0]) , int(x[80].split('+')[0]),
		                  int(x[81].split('+')[0]), int(x[82].split('+')[0]), int(x[83].split('+')[0]),int(x[84].split('+')[0]), int(x[85].split('+')[0]), int(x[86].split('+')[0]) , int(x[87].split('+')[0])))
	# Convertimos el rdd a df					  
	df = rdd.toDF(["Position","Crossing","Finishing","HeadingAccuracy","ShortPassing","Volleys","Dribbling","Curve",
                   "FKAccuracy", "LongPassing","BallControl","Acceleration","SprintSpeed","Agility","Reactions",
                   "Balance","ShotPower","Jumping","Stamina","Strength","LongShots","Aggression",
                   "Interceptions","Positioning","Vision","Penalties","Composure","Marking","StandingTackle",
                   "SlidingTackle","GKDiving","GKHandling","GKKicking","GKPositioning","GKReflexes"])
	return df


# Analizar con los features mas representativos para el modelo
def leer_df_categoricos():
	conf = SparkConf().setAppName("Tarea4").setMaster("local")
	sc = SparkContext(conf=conf)

	sqlContext = SQLContext(sc)
	
	# Creacion de rdd
	rdd = sqlContext.read.csv("data.csv", header=True).rdd
	
	# Filtramos los datos vacios
	rdd = rdd.filter(
		lambda x: (		   x[21] != None and x[21] != '' and x[55] != None and x[57] != None and x[63] != None and \
						   x[71] != None and x[82] != None and x[87] != None and x[54] != None and x[66] != None and x[59] != None and x[65] != None  ))

	# Features mas representativos
	rdd = rdd.map(
        lambda x: (tipo_posiciones((x[21])) ,int(x[55].split('+')[0]), int(x[57].split('+')[0]), int(x[63].split('+')[0]),
		int(x[71].split('+')[0]),int(x[82].split('+')[0]), int(x[87].split('+')[0]),int(x[54].split('+')[0]),int(x[66].split('+')[0]),int(x[59].split('+')[0]), int(x[65].split('+')[0])  ))
	df = rdd.toDF(["Position", "Finishing", "ShortPassing", "BallControl", "Stamina", "SlidingTackle", "GKReflexes", "Crossing", "Agility", "Dribbling", "SprintSpeed"])

	return df

# Seleccionar los features mas representativos para el modelo
def feature_selection(df):
	# Creamos vectorassembler
	assembler = VectorAssembler(
		inputCols=["Crossing","Finishing","HeadingAccuracy","ShortPassing","Volleys","Dribbling","Curve",
                   "FKAccuracy", "LongPassing","BallControl","Acceleration","SprintSpeed","Agility","Reactions",
                   "Balance","ShotPower","Jumping","Stamina","Strength","LongShots","Aggression",
                   "Interceptions","Positioning","Vision","Penalties","Composure","Marking","StandingTackle",
                   "SlidingTackle","GKDiving","GKHandling","GKKicking","GKPositioning","GKReflexes"],
		outputCol="features")
	df = assembler.transform(df)

	# Vectorindexer   
	indexer = VectorIndexer(
		inputCol="features", 
		outputCol="indexedFeatures")
	
	df = indexer.fit(df).transform(df)

	# Prueba ChiSquare
	selector = ChiSqSelector(
		numTopFeatures=10,
		featuresCol="indexedFeatures",
		labelCol="Position",
		outputCol="selectedFeatures")
	resultado = selector.fit(df).transform(df)
	resultado.select("features", "selectedFeatures").show()

def entrenamiento(df):
	# Vectorizo
	df = df.select("Finishing", "ShortPassing", "BallControl", "Stamina", "SlidingTackle", "GKReflexes", "Crossing", "Agility", "Position", "Dribbling", "SprintSpeed")
	assembler = VectorAssembler(
		inputCols=["Finishing", "ShortPassing", "BallControl", "Stamina", "SlidingTackle", "GKReflexes", "Crossing", "Agility", "Dribbling", "SprintSpeed"],
		outputCol="features")
	df = assembler.transform(df)

	# Dividir nuestro dataset
	(training_df, test_df) = df.randomSplit([0.7, 0.3])

	# Entrenamiento
	entrenador = DecisionTreeClassifier(
		labelCol="Position", 
		featuresCol="features")

	# Creacion de pipeline
	pipeline = Pipeline(stages=[entrenador])
    # Se entrena el modelo
	model = pipeline.fit(training_df)

	# Prediccion
	predictions_df = model.transform(test_df)

	# Evaluador --> Accuracy
	evaluator = MulticlassClassificationEvaluator(
		labelCol="Position",
		predictionCol="prediction",
		metricName="accuracy")

	# Exactitud
	exactitud = evaluator.evaluate(predictions_df)
	print("Exactitud: {}".format(exactitud))

def main():
	#df = leer_df()
	df = leer_df_categoricos()
	#feature_selection(df)
	entrenamiento(df)
	
if __name__ == "__main__":
	main()
