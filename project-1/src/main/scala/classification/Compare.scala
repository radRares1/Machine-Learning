package classification

import classification.LogReg.matrix
import org.apache.log4j._
import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.feature.{OneHotEncoderEstimator, StringIndexer, VectorAssembler}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{avg, col, lit}

object Compare extends App {


  def getPrediction(df:DataFrame):Double = {
    val Array(training, test) = df.randomSplit(Array(0.7, 0.3), seed = 12345)

    import org.apache.spark.ml.Pipeline

    val decisionTree = new DecisionTreeClassifier()

    val genderIndexer = new StringIndexer().setInputCol("Sex").setOutputCol("SexIndex")
    val embarkIndexer = new StringIndexer().setInputCol("Embarked").setOutputCol("EmbarkIndex")

    val genderEncoder = new OneHotEncoderEstimator().setInputCols(Array("SexIndex")).setOutputCols(Array("SexVec"))
    val embarkEncoder = new OneHotEncoderEstimator().setInputCols(Array("EmbarkIndex")).setOutputCols(Array("EmbarkVec"))

    // Assemble everything together to be ("label","features") format
    val assembler = new VectorAssembler()
      .setInputCols(Array("Pclass", "SexVec", "Age", "SibSp", "Parch", "Fare", "EmbarkVec"))
      .setOutputCol("features")

    val pipeline = new Pipeline().setStages(Array(genderIndexer, embarkIndexer, genderEncoder, embarkEncoder, assembler, decisionTree))
    val model = pipeline.fit(training)

    val results = model.transform(test)
    import org.apache.spark.mllib.evaluation.MulticlassMetrics
    import spark.implicits._

    val predictionAndLabels = results
      .select(col("prediction"), col("label"))
      .as[(Double, Double)].rdd

    val metrics = new MulticlassMetrics(predictionAndLabels)
    val matrix = metrics.confusionMatrix
    val prediction = matrix(0,0)/(matrix(0,0)+matrix(0,1))
    prediction
  }

  Logger.getLogger("org").setLevel(Level.ERROR)

  // Spark Session
  val conf = new SparkConf()

  lazy val spark = SparkSession
    .builder()
    .appName("LogReg")
    .master("local[*]")
    .config(conf)
    .getOrCreate()

  // Use Spark to read in the Titanic csv file.
  val data = spark
    .read
    .option("header", "true")
    .option("inferSchema", "true")
    .format("csv")
    .load("src/main/scala/classification/titanic.csv")

  data.printSchema()

  val dataWithNulls = {
    data.select(data("Survived")
      .as("label"), col("Pclass"), col("Sex"), col("Age"), col("SibSp"),
      col("Parch"), col("Fare"), col("Embarked"))
  }
  val cleanData = dataWithNulls.na.drop()


  cleanData.printSchema()

  val avgs = cleanData.select(avg(col("Pclass")),
    avg(col("Age")),avg(col("SibSp")),avg(col("Parch")),
    avg(col("Fare")))

  avgs.show
  val avgsList = avgs.collect.toList.head.toSeq.toList
  println(avgs.collect().toList.head.toSeq.toList)

  val avgPClass = cleanData.drop(col("Pclass")).withColumn("Pclass",lit(avgsList(0)))
  val avgAge = cleanData.drop(col("Age")).withColumn("Age",lit(avgsList(1)))
  val avgSibSp = cleanData.drop(col("SibSp")).withColumn("SibSp",lit(avgsList(2)))
  val avgParch = cleanData.drop(col("Parch")).withColumn("Parch",lit(avgsList(3)))
  val avgFare = cleanData.drop(col("Fare")).withColumn("Fare",lit(avgsList(4)))

  val list = List(avgPClass, avgAge, avgSibSp, avgParch, avgFare)
  val base = getPrediction(cleanData)

  val differences = list.map(getPrediction).map(prediction => math.abs(base - prediction)).zipWithIndex.map {
    case (diff, 0) => (diff,"Pclass")
    case (diff, 1) => (diff,"Age")
    case (diff, 2) => (diff,"SibSp")
    case (diff, 3) => (diff,"Parch")
    case (diff, 4) => (diff,"Fare")
  }

  println(differences)

}