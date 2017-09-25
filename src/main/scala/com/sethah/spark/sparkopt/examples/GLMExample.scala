package com.sethah.spark.sparkopt.examples

import com.sethah.spark.sparkopt.ml.BaseAlgorithm
import com.sethah.spark.sparkopt.ml.optim.loss.{GLMLoss, L1Regularization, L2Regularization, SeparableDiffFun}
import com.sethah.spark.sparkopt.ml.optim.minimizers._
import org.apache.spark.ml.{InstanceWrapper, ModelFactory}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.regression.{FamilyAndLink, GLMWrapper, GeneralizedLinearRegression, LinearRegressionModel, LinearRegression => SparkLinearRegression}
import org.apache.spark.ml.classification.{BinaryLogisticRegressionTrainingSummary, LogisticRegressionModel, LogisticRegression => SparkLogisticRegression}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.rdd.RDD
import scopt.OptionParser
import sun.java2d.loops.FillRect.General

object GLMExample {

  private[this] case class Params(
    family: String = "binomial",
    link: String = "logit",
    minimizer: String = "lbfgs",
    l1Reg: Double = 0.0,
    l2Reg: Double = 0.0,
    fitIntercept: Boolean = true,
    trainPath: Option[String] = None)

  private[this] object Params {
    def parseArgs(args: Array[String]): Params = {
      val params = new OptionParser[Params]("train an existing model") {
        opt[String]("family")
          .text("glm family")
          .action((x, c) => {
            require(Seq("binomial", "gaussian", "poisson").contains(x),
              s"family $x is not supported yet")
            c.copy(family = x)
          })
        opt[String]("link")
          .text("glm link function")
          .action((x, c) => {
            require(Seq("logit", "log", "identity").contains(x), s"link $x is not supported yet")
            c.copy(link = x)
          })
        opt[String]("minimizer")
          .text("minimizer")
          .action((x, c) => {
            require(Seq("lbfgs", "admm", "owlqn").contains(x), s"minimizer $x is not supported yet")
            c.copy(minimizer = x)
          })
        opt[String]("trainPath")
          .text("path for training data")
          .action((x, c) => c.copy(trainPath = Some(x)))
        opt[Double]("l2Reg")
          .text("l2 regularization")
          .action((x, c) => c.copy(l2Reg = x))
        opt[Double]("l1Reg")
          .text("l1 regularization")
          .action((x, c) => c.copy(l1Reg = x))
      }.parse(args, Params()).get
      require(params.trainPath.isDefined, "must supply train data path")
      params
    }
  }

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("GLMOptimizationExample")
      .getOrCreate()
    org.apache.log4j.Logger.getRootLogger.setLevel(org.apache.log4j.Level.WARN)
    val params = Params.parseArgs(args)
    val df = spark.read.parquet(params.trainPath.get)

    // choose a GLM family and link function
    val family = GLMWrapper.familyFromString(params.family)
    val link = GLMWrapper.linkFromString(params.link)
    val familyAndLink = new FamilyAndLink(family, link)
    val logLike = GLMWrapper.getLogLikelihood(family)

    val numFeatures = df.select("features").first().getAs[Vector](0).size
    val fitIntercept = params.fitIntercept
    val numFeaturesPlusIntercept = if (fitIntercept) numFeatures + 1 else numFeatures
    val initialCoefficients = Vectors.zeros(numFeaturesPlusIntercept)

    // don't apply regularization to the intercept
    def indexToReg(reg: Double)(idx: Int): Double = {
      if (idx >= 0 && idx < numFeatures) reg else 0.0
    }
    val l2Reg = new L2Regularization(indexToReg(params.l2Reg))
    val l1Reg = new L1Regularization(indexToReg(params.l1Reg))

    // supply the base solver with an optimizer, loss function, and initial parameters
    val instanceFunc = GLMLoss.apply(_: InstanceWrapper.Instance,
      familyAndLink, fitIntercept, logLike)
    val minimizer = minimizerFromString(params.minimizer)
    val base = new BaseAlgorithm(instanceFunc)
      .setInitialParams(initialCoefficients)
      .setMinimizer(minimizer)
      .setL2Reg(l2Reg)
      .setL1Reg(l1Reg)

    // fit and evaluate
    val baseModel = base.fit(df)
    val model = ModelFactory.createGeneralizedLinearRegression(baseModel.uid, baseModel,
      fitIntercept, familyAndLink)
    val summary = model.evaluate(df)

    val regParam = params.l1Reg + params.l2Reg
    val elasticNetParam = if (regParam == 0.0) 0.0 else params.l1Reg / regParam
    val sparkEstimator = new SparkLogisticRegression()
      .setRegParam(regParam)
      .setElasticNetParam(elasticNetParam)
      .setStandardization(false)
    val sparkModel = sparkEstimator.fit(df)
    println(model.coefficients)
    println(model.intercept)
    println(summary.aic)
  }

  def minimizerFromString(
     minimizer: String): IterativeMinimizer[Vector,
    SeparableDiffFun[RDD], IterativeMinimizerState[Vector]] = {
    minimizer match {
      case "lbfgs" => new LBFGS()
      case "admm" => new ConsensusADMM(new LBFGS().setMaxIter(20))
      case "owlqn" => new OWLQN()
      case _ => throw new IllegalArgumentException(s"minimizer $minimizer not supported")
    }
  }

}
