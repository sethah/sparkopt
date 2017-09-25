package org.apache.spark.ml.regression

object GLMWrapper {
  val binomial = GeneralizedLinearRegression.Binomial
  val gaussian = GeneralizedLinearRegression.Gaussian
  val poisson = GeneralizedLinearRegression.Poisson

  val logit = GeneralizedLinearRegression.Logit
  val identity = GeneralizedLinearRegression.Identity
  val log = GeneralizedLinearRegression.Log

  def familyFromString(family: String): GeneralizedLinearRegression.Family = {
    family match {
      case "binomial" => binomial
      case "poisson" => poisson
      case "gaussian" => gaussian
      case _ => throw new IllegalArgumentException(s"family $family not supported")
    }
  }

  def linkFromString(link: String): GeneralizedLinearRegression.Link = {
    link match {
      case "logit" => logit
      case "log" => log
      case "identity" => identity
      case _ => throw new IllegalArgumentException(s"link $link not supported")
    }
  }

  def getLogLikelihood(
    family: GeneralizedLinearRegression.Family): (Double, Double, Double) => Double = {
    (y: Double, mu: Double, w: Double) => {
      family match {
        case `binomial` => (y * math.log(mu) + (1 - y) * math.log(1 - mu)) * w
        case `poisson` => -w * (y * math.log(mu) - mu)
        case `gaussian` => 0.5 * w * ((y - mu) * (y - mu) + math.log(2 * math.Pi))
        case _ => throw new IllegalArgumentException(s"family $family not supported")
      }
    }
  }
}

class FamilyAndLink(family: GeneralizedLinearRegression.Family,
                    link: GeneralizedLinearRegression.Link)
  extends GeneralizedLinearRegression.FamilyAndLink(family, link)

