package org.apache.spark.ml.regression

object GLMWrapper {
  val binomial = GeneralizedLinearRegression.Binomial
  val logit = GeneralizedLinearRegression.Logit

  def familyFromString(family: String): GeneralizedLinearRegression.Family = {
    family match {
      case "binomial" => binomial
      case _ => throw new IllegalArgumentException(s"family $family not supported")
    }
  }

  def linkFromString(link: String): GeneralizedLinearRegression.Link = {
    link match {
      case "logit" => logit
      case _ => throw new IllegalArgumentException(s"link $link not supported")
    }
  }

  def getLogLikelihood(
    family: GeneralizedLinearRegression.Family): (Double, Double, Double) => Double = {
    (y: Double, mu: Double, w: Double) => {
      family match {
        case GeneralizedLinearRegression.Binomial =>
            (y * math.log(mu) + (1 - y) * math.log(1 - mu)) * w
        case _ => throw new IllegalArgumentException(s"family $family not supported")
      }
    }

  }
}

class FamilyAndLink(family: GeneralizedLinearRegression.Family,
                    link: GeneralizedLinearRegression.Link)
  extends GeneralizedLinearRegression.FamilyAndLink(family, link)

