/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.sethah.spark.sparkopt.ml.optim.loss

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.InstanceWrapper
import org.apache.spark.ml.linalg.BLASWrapper.{instance => BLAS}
import org.apache.spark.ml.regression.{FamilyAndLink, GLMWrapper}
import org.apache.spark.mllib.util.MLUtils

trait InstanceLoss extends DiffFun[Vector] {

  def isIntercept(index: Int): Boolean

}

case class BinomialLoss(
    instance: InstanceWrapper.tpe,
    fitIntercept: Boolean) extends InstanceLoss {

  private val numFeaturesPlusIntercept = instance.features.size + (if (fitIntercept) 1 else 0)
  override def weight: Double = instance.weight

  override def doCompute(x: Vector): (Double, Vector) = {
    val grad = Vectors.zeros(x.size)
    val loss = computeInPlace(x, grad)
    (loss, grad)
  }

  def doComputeInPlace(x: Vector, grad: Vector): Double = {
    val localCoefficients = x.toArray
    val margin = - {
      var sum = 0.0
      instance.features.foreachActive { (index, value) =>
        if (value != 0.0) {
          sum += localCoefficients(index) * value
        }
      }
      if (fitIntercept) sum += localCoefficients(numFeaturesPlusIntercept - 1)
      sum
    }

    val multiplier = weight * (1.0 / (1.0 + math.exp(margin)) - instance.label)

    val localGradientArray = grad.toArray

    instance.features.foreachActive { (index, value) =>
      if (value != 0.0) {
        localGradientArray(index) += multiplier * value
      }
    }

    if (fitIntercept) {
      localGradientArray(numFeaturesPlusIntercept - 1) += multiplier
    }

    if (instance.label > 0) {
      // The following is equivalent to log(1 + exp(margin)) but more numerically stable.
      weight * log1pExp(margin)
    } else {
      weight * (log1pExp(margin) - margin)
    }
  }

  def log1pExp(x: Double): Double = {
    if (x > 0) {
      x + math.log1p(math.exp(-x))
    } else {
      math.log1p(math.exp(x))
    }
  }

  override def isIntercept(index: Int): Boolean = {
    fitIntercept && index == instance.features.size - 1
  }
}

case class StdBinomialLoss(
                            instance: InstanceWrapper.tpe,
                            fitIntercept: Boolean,
                            featuresStd: Broadcast[Array[Double]]) extends InstanceLoss {

  private val numFeaturesPlusIntercept = instance.features.size + (if (fitIntercept) 1 else 0)
  override def weight: Double = instance.weight

  override def doCompute(x: Vector): (Double, Vector) = {
    throw new NotImplementedError("not implemented!")
  }

  def doComputeInPlace(x: Vector, grad: Vector): Double = {
    val localFeaturesStd = featuresStd.value
    val localCoefficients = x.toArray
    val margin = - {
      var sum = 0.0
      instance.features.foreachActive { (index, value) =>
        if (localFeaturesStd(index) != 0.0 && value != 0.0) {
          sum += localCoefficients(index) * value / localFeaturesStd(index)
        }
      }
      if (fitIntercept) sum += localCoefficients(numFeaturesPlusIntercept - 1)
      sum
    }

    val multiplier = weight * (1.0 / (1.0 + math.exp(margin)) - instance.label)

    val localGradientArray = grad.toArray

    instance.features.foreachActive { (index, value) =>
      if (localFeaturesStd(index) != 0.0 && value != 0.0) {
        localGradientArray(index) += multiplier * value / localFeaturesStd(index)
      }
    }

    if (fitIntercept) {
      localGradientArray(numFeaturesPlusIntercept - 1) += multiplier
    }

    if (instance.label > 0) {
      // The following is equivalent to log(1 + exp(margin)) but more numerically stable.
      weight * log1pExp(margin)
    } else {
      weight * (log1pExp(margin) - margin)
    }
  }

  def log1pExp(x: Double): Double = {
    if (x > 0) {
      x + math.log1p(math.exp(-x))
    } else {
      math.log1p(math.exp(x))
    }
  }

  override def isIntercept(index: Int): Boolean = {
    fitIntercept && index == instance.features.size - 1
  }

}

case class SquaredLoss(instance: InstanceWrapper.tpe, fitIntercept: Boolean)
  extends InstanceLoss {

  override val weight: Double = instance.weight

  override def doCompute(x: Vector): (Double, Vector) = {
    throw new NotImplementedError("not implemented!")
  }

  def doComputeInPlace(x: Vector, grad: Vector): Double = {
    val localCoefficients = x.toArray
    val pred = {
      var sum = 0.0
      instance.features.foreachActive { (index, value) =>
        if (value != 0.0) {
          sum += localCoefficients(index) * value
        }
      }
      if (fitIntercept) sum += localCoefficients(x.size - 1)
      sum
    }
    val err = pred - instance.label
    if (err != 0) {
      val localGradientSumArray = grad.toArray
      instance.features.foreachActive { (index, value) =>
        if (value != 0.0) {
          localGradientSumArray(index) += instance.weight * err * value
        }
      }
      if (fitIntercept) localGradientSumArray(x.size - 1) += instance.weight * err
    }
    0.5 * instance.weight * err * err
  }

  override def isIntercept(index: Int): Boolean = {
    fitIntercept && index == instance.features.size - 1
  }
}
case class StdSquaredLoss(
    instance: InstanceWrapper.tpe,
    fitIntercept: Boolean,
    labelStd: Double,
    featuresStd: Broadcast[Array[Double]]) extends InstanceLoss {

  override val weight: Double = instance.weight

  override def doCompute(x: Vector): (Double, Vector) = {
    throw new NotImplementedError("not implemented!")
  }

  def doComputeInPlace(x: Vector, grad: Vector): Double = {
    val localFeaturesStd = featuresStd.value
    val localCoefficients = x.toArray
    val pred = {
      var sum = 0.0
      instance.features.foreachActive { (index, value) =>
        if (localFeaturesStd(index) != 0.0 && value != 0.0) {
          sum += localCoefficients(index) * value
        }
      }
      if (fitIntercept) sum += localCoefficients(x.size - 1)
      sum
    }
    val err = pred - instance.label / labelStd
    if (err != 0) {
      val localGradientSumArray = grad.toArray
      instance.features.foreachActive { (index, value) =>
        val fStd = localFeaturesStd(index)
        if (fStd != 0.0 && value != 0.0) {
          localGradientSumArray(index) += instance.weight * err * value / fStd
        }
      }
      if (fitIntercept) localGradientSumArray(x.size - 1) += instance.weight * err
    }
    0.5 * instance.weight * err * err
  }

  override def isIntercept(index: Int): Boolean = {
    fitIntercept && index == instance.features.size - 1
  }
}

//case class StdGLMLoss(
//                       instance: Instance,
//                       familyAndLink: FamilyAndLink,
//                       fitIntercept: Boolean,
//                       featuresStd: Broadcast[Array[Double]]) extends DiffFun[Vector] {
//
//  private val numFeaturesPlusIntercept = instance.features.size + (if (fitIntercept) 1 else 0)
//  override def weight: Double = instance.weight
//
//  override def doCompute(x: Vector): (Double, Vector) = {
//    throw new NotImplementedError("not implemented!")
//  }
//
//  def doComputeInPlace(x: Vector, grad: Vector): Double = {
//    val localFeaturesStd = featuresStd.value
//    val localCoefficients = x.toArray
//    val eta = {
//      var sum = 0.0
//      instance.features.foreachActive { (index, value) =>
//        if (localFeaturesStd(index) != 0.0 && value != 0.0) {
//          sum += localCoefficients(index) * value
//        }
//      }
//      if (fitIntercept) sum += localCoefficients(numFeaturesPlusIntercept - 1)
//      sum
//    }
//    val mu = familyAndLink.fitted(eta)
//    val error = mu - instance.label
//    val mult = error / (familyAndLink.link.deriv(mu) * familyAndLink.family.variance(mu))
//
//    if (error != 0) {
//      val localGradientSumArray = grad.toArray
//      instance.features.foreachActive { (index, value) =>
//        val fStd = localFeaturesStd(index)
//        if (fStd != 0.0 && value != 0.0) {
//          localGradientSumArray(index) += weight * mult * value / fStd
//        }
//      }
//      if (fitIntercept) localGradientSumArray(instance.features.size) += weight * mult
//    }
//    val ll = -familyAndLink.family.logLikelihood(instance.label, mu, weight)
//    //    println(ll)
//    ll
//  }
//}

case class GLMLoss(
                    instance: InstanceWrapper.tpe,
                    familyAndLink: FamilyAndLink,
                    fitIntercept: Boolean,
                  logLikelihood: (Double, Double, Double) => Double) extends InstanceLoss {

  private val numFeaturesPlusIntercept = instance.features.size + (if (fitIntercept) 1 else 0)
  override def weight: Double = instance.weight

  override def doCompute(x: Vector): (Double, Vector) = {
    throw new NotImplementedError("not implemented!")
  }

  def doComputeInPlace(x: Vector, grad: Vector): Double = {
    val localCoefficients = x.toArray
    val eta = {
      var sum = 0.0
      instance.features.foreachActive { (index, value) =>
        if (value != 0.0) {
          sum += localCoefficients(index) * value
        }
      }
      if (fitIntercept) sum += localCoefficients(numFeaturesPlusIntercept - 1)
      sum
    }
    val mu = familyAndLink.fitted(eta)
    val error = mu - instance.label
    val mult = error / (familyAndLink.link.deriv(mu) * familyAndLink.family.variance(mu))

    if (error != 0) {
      val localGradientSumArray = grad.toArray
      instance.features.foreachActive { (index, value) =>
        if (value != 0.0) {
          localGradientSumArray(index) += weight * mult * value
        }
      }
      if (fitIntercept) localGradientSumArray(instance.features.size) += weight * mult
    }
    -logLikelihood(instance.label, mu, weight)
  }

  override def isIntercept(index: Int): Boolean = {
    fitIntercept && index == instance.features.size - 1
  }
}

