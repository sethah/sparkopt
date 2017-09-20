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
package com.sethah.spark.sparkopt.ml.optim.minimizers

import com.sethah.spark.sparkopt.ml.optim.loss._
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.linalg.BLASWrapper.{instance => BLAS}
import org.apache.spark.ml.param._
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.storage.StorageLevel

import scala.collection.mutable

trait ConsensusADMMParams extends Params {

  final val maxIter: IntParam = new IntParam(this, "maxIter", "Maximum number of iterations.",
    ParamValidators.gt(0))

  def getMaxIter: Int = $(maxIter)

  final val rho: DoubleParam = new DoubleParam(this, "rho", "The initial step size for ADMM.",
    ParamValidators.gt(0.0))

  def getRho: Double = $(rho)

  final val primalTol: DoubleParam = new DoubleParam(this, "primalTol", "Primal tolerance to use" +
    "for ADMM convergence.", ParamValidators.gt(0.0))

  def getPrimalTol: Double = $(primalTol)

  final val dualTol: DoubleParam = new DoubleParam(this, "dualTol", "Dual tolerance to use " +
    "for ADMM convergence.", ParamValidators.gt(0.0))

  def getDualTol: Double = $(dualTol)

}

class ConsensusADMM(partitionMinimizer: IterativeMinimizer[Vector,
  DiffFun[Vector], IterativeMinimizerState[Vector]], override val uid: String)
  extends IterativeMinimizer[Vector, SeparableDiffFun[RDD], ConsensusADMM.ADMMState[Vector]]
    with ConsensusADMMParams {

  import ConsensusADMM._

  def this(partitionMinimizer: IterativeMinimizer[Vector,
    DiffFun[Vector], IterativeMinimizerState[Vector]]) = this(partitionMinimizer,
    Identifiable.randomUID("cadmm"))

  type State = ADMMState[Vector]

  def setRho(value: Double): this.type = set(rho, value)
  setDefault(rho -> 0.001)

  def setPrimalTol(value: Double): this.type = set(primalTol, value)
  setDefault(primalTol -> 1e-4)

  def setDualTol(value: Double): this.type = set(dualTol, value)
  setDefault(dualTol -> 1e-4)

  def setMaxIter(value: Int): this.type = set(maxIter, value)
  setDefault(maxIter -> 100)

  def initialState(lossFunction: SeparableDiffFun[RDD], initialParams: Vector): State = {
    val firstLoss = lossFunction.loss(initialParams)
    val initialModels = lossFunction.losses.mapPartitions { it =>
      Iterator.single(ADMMIterationState(firstLoss, initialParams.copy, initialParams.copy, 1,
        Double.MaxValue))
    }
    ADMMState(initialParams, 0, firstLoss, Double.MaxValue, Double.MaxValue, $(rho), initialModels)
  }

  def iterations(lossFunction: SeparableDiffFun[RDD],
                 initialParameters: Vector): Iterator[State] = {
    val tauIncrease = 2.0
    val tauDecrease = 2.0
    val mu = 100.0
    val numFeatures = initialParameters.size
    val iters = Iterator.iterate(initialState(lossFunction, initialParameters)) { state =>
      val oldParams = state.params
      val iter = state.iter
      val newSubModels = lossFunction.losses
        .zipPartitions(state.subModels, preservesPartitioning = true) { case (losses, models) =>
          val modelState = models.next()
          val (x, u) = (modelState.x, modelState.u)
          val residual = if (iter == 0) Double.MaxValue else Vectors.sqdist(x, oldParams)
          BLAS.axpy(1.0, x, u)
          BLAS.axpy(-1.0, oldParams, u)
          val problems = losses.toIterable
          val partitionLoss = new SeparableDiffFun(problems, lossFunction.getAggregator,
            List.empty[EnumeratedRegularization[Vector]])
          val admmLoss = new ConsensusADMMLoss(partitionLoss, oldParams, u, state.rho)
          val optIterations = partitionMinimizer.iterations(admmLoss, oldParams)
          var lastIter: IterativeMinimizerState[Vector] = null
          val arrayBuilder = mutable.ArrayBuilder.make[Double]
          while (optIterations.hasNext) {
            lastIter = optIterations.next()
            arrayBuilder += lastIter.loss
          }
          Iterator.single(ADMMIterationState(lastIter.loss, u, lastIter.params, 1, residual))
        }.persist(StorageLevel.MEMORY_AND_DISK)

      val seqOp = (acc: (Double, Vector, Long, Double), model: ADMMIterationState) => {
        val res: Vector = acc._2.toDense
        BLAS.axpy(1.0, model.u, res)
        BLAS.axpy(1.0, model.x, res)
        (acc._1 + model.loss, res, acc._3 + model.count, acc._4 + model.residual)
      }
      val combOp = (acc1: (Double, Vector, Long, Double), acc2: (Double, Vector, Long, Double)) => {
        val res = acc2._2.toDense
        BLAS.axpy(1.0, acc1._2, res)
        (acc1._1 + acc2._1, res, acc1._3 + acc2._3, acc1._4 + acc2._4)
      }
      val (loss, zSum, count, sqPrimalResidual) = newSubModels.treeAggregate(
        (0.0, Vectors.sparse(numFeatures, Array.emptyIntArray, Array.emptyDoubleArray), 0L, 0.0))(
        seqOp, combOp)

      state.subModels.unpersist()

      val regLoss = lossFunction.regularizers match {
        case List(l2: L2Regularization) =>
          val zSumArray = zSum.toArray
          zSumArray.indices.foreach { j =>
            zSumArray(j) *= state.rho / (count.toDouble * (l2.regFunc(j) + state.rho))
          }
          l2.apply(zSum)
        case List(l1: L1Regularization) =>
          val zSumArray = zSum.toArray
          zSumArray.indices.foreach { j =>
            val kappa = l1.regFunc(j) / state.rho
            zSumArray(j) = ConsensusADMM.shrinkage(zSumArray(j) / count.toDouble, kappa)
          }
          l1.apply(zSum)
        case List(l1: L1Regularization, l2: L2Regularization) =>
          val zSumArray = zSum.toArray
          zSumArray.indices.foreach { j =>
            val kappa = l1.regFunc(j) / (l2.regFunc(j) + state.rho)
            val mult = state.rho / (count.toDouble * (l2.regFunc(j) + state.rho))
            zSumArray(j) = ConsensusADMM.shrinkage(zSumArray(j) * mult, kappa)
          }
          l2.apply(zSum) + l1.apply(zSum)
        case _ =>
          BLAS.scal(1 / count.toDouble, zSum)
          0.0
      }
      val primalResidual = math.sqrt(sqPrimalResidual)
      val dualResidual = state.rho * math.sqrt(count * Vectors.sqdist(oldParams, zSum))
      val newRho = if (primalResidual > mu * dualResidual) {
        tauIncrease * state.rho
      } else if (dualResidual > mu * primalResidual) {
        state.rho / tauDecrease
      } else {
        state.rho
      }
      ADMMState(zSum, state.iter + 1, loss + regLoss, primalResidual, dualResidual, newRho,
        newSubModels)
    }.takeWhile { state =>
      println(s"iter: ${state.iter} old loss: ${state.loss}, rho: ${state.rho}, " +
        s"primalResid: ${state.primalResidual}, dualResid: ${state.dualResidual}, " +
        s"int: ${state.params(state.params.size - 1)}")
      state.dualResidual > $(dualTol) && state.primalResidual > $(primalTol) && state.iter < getMaxIter
    }
    iters
  }

  override def copy(extra: ParamMap): this.type = defaultCopy(extra)

}

object ConsensusADMM {
  case class ADMMState[+T](
      params: T,
      iter: Int,
      loss: Double,
      primalResidual: Double,
      dualResidual: Double,
      rho: Double,
      subModels: RDD[ADMMIterationState]) extends IterativeMinimizerState[T]

  case class ADMMIterationState(loss: Double, u: Vector, x: Vector, count: Long, residual: Double)

  private def shrinkage(x: Double, kappa: Double): Double = {
    math.max(0, x - kappa) - math.max(0, x + kappa) + kappa + x
  }
}

