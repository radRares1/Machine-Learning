package classification
import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{OneHotEncoderEstimator, StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col


import org.apache.log4j._

object LogReg extends App {

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

  ///////////////////////
  /// Display Data /////
  /////////////////////
  val colnames = data.columns
  val firstrow = data.head(1)(0)
  println("\n")
  println("Example Data Row")
  for (ind <- Range(1, colnames.length)) {
    println(colnames(ind))
    println(firstrow(ind))
    println("\n")
  }

  ////////////////////////////////////////////////////
  //// Setting Up DataFrame for Machine Learning ////
  //////////////////////////////////////////////////

  // Grab only the columns we want
  val dataWithNulls = {
    data.select(data("Survived")
      .as("label"), col("Pclass"), col("Sex"), col("Age"), col("SibSp"),
      col("Parch"), col("Fare"), col("Embarked"))
  }
  val cleanData = dataWithNulls.na.drop()

  // A few things we need to do before Spark can accept the data!
  // We need to deal with the Categorical columns


  // Import VectorAssembler and Vectors



  // Deal with Categorical Columns

  val genderIndexer = new StringIndexer().setInputCol("Sex").setOutputCol("SexIndex")
  val embarkIndexer = new StringIndexer().setInputCol("Embarked").setOutputCol("EmbarkIndex")

  val genderEncoder = new OneHotEncoderEstimator().setInputCols(Array("SexIndex")).setOutputCols(Array("SexVec"))
  val embarkEncoder = new OneHotEncoderEstimator().setInputCols(Array("EmbarkIndex")).setOutputCols(Array("EmbarkVec"))

  // Assemble everything together to be ("label","features") format
  val assembler = (new VectorAssembler()
    .setInputCols(Array("Pclass", "SexVec", "Age", "SibSp", "Parch", "Fare", "EmbarkVec"))
    .setOutputCol("features"))


  ////////////////////////////
  /// Split the Data ////////
  //////////////////////////
  val Array(training, test) = cleanData.randomSplit(Array(0.7, 0.3), seed = 12345)


  ///////////////////////////////
  // Set Up the Pipeline ///////
  /////////////////////////////

  import org.apache.spark.ml.Pipeline

  val lr = new LogisticRegression()

  val pipeline = new Pipeline().setStages(Array(genderIndexer, embarkIndexer, genderEncoder, embarkEncoder, assembler, lr))

  // Fit the pipeline to training documents.
  val model = pipeline.fit(training)

  // Get Results on Test Set
  val results = model.transform(test)

  results.show(false)

  //// MODEL EVALUATION ////

  import org.apache.spark.mllib.evaluation.MulticlassMetrics
  import spark.implicits._

  val predictionAndLabels = results
    .select(col("prediction"), col("label"))
    .as[(Double, Double)].rdd

  // Instantiate metrics object
  val metrics = new MulticlassMetrics(predictionAndLabels)

  val matrix = metrics.confusionMatrix
  val prediction = matrix(0,0)/(matrix(0,0)+matrix(0,1))
  println(prediction)
  println(metrics.accuracy)

  // Confusion matrix
  println("Logistic regression confusion matrix:")
  println(metrics.confusionMatrix)
}