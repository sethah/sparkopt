package org.apache.spark.ml

import com.sethah.spark.sparkopt.ml.BaseAlgorithmModel
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.regression.{FamilyAndLink, GeneralizedLinearRegressionModel, LinearRegressionModel}

/**
 * Wrapper methods to hack into Spark's internals for creating models.
 */
object ModelFactory {

  def createBinaryLogisticRegression(
      uid: String,
      baseModel: BaseAlgorithmModel,
      fitIntercept: Boolean): LogisticRegressionModel = {
    val allCoef = baseModel.coefficients.toArray
    val coef = if (fitIntercept) Vectors.dense(allCoef.init) else Vectors.dense(allCoef)
    val intercept = if (fitIntercept) allCoef.last else 0.0
    new LogisticRegressionModel(uid, coef, intercept).copy(baseModel.extractParamMap())
  }

  def createLinearRegression(
      uid: String,
      baseModel: BaseAlgorithmModel,
      fitIntercept: Boolean): LinearRegressionModel = {
    val allCoef = baseModel.coefficients.toArray
    val coef = if (fitIntercept) Vectors.dense(allCoef.init) else Vectors.dense(allCoef)
    val intercept = if (fitIntercept) allCoef.last else 0.0
    new LinearRegressionModel(uid, coef, intercept).copy(baseModel.extractParamMap())
  }

  def createGeneralizedLinearRegression(
      uid: String,
      baseModel: BaseAlgorithmModel,
      fitIntercept: Boolean,
      familyAndLink: FamilyAndLink): GeneralizedLinearRegressionModel = {
    val allCoef = baseModel.coefficients.toArray
    val coef = if (fitIntercept) Vectors.dense(allCoef.init) else Vectors.dense(allCoef)
    val intercept = if (fitIntercept) allCoef.last else 0.0
    val glr = new GeneralizedLinearRegressionModel(uid, coef, intercept)
      .copy(baseModel.extractParamMap())
    val extra = new ParamMap()
      .put(glr.family, familyAndLink.family.name)
      .put(glr.link, familyAndLink.link.name)
    glr.copy(extra)
  }

}
