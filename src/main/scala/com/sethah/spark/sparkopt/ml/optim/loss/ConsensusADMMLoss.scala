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

import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.linalg.BLASWrapper.{instance => BLAS}

class ConsensusADMMLoss[F <: DiffFun[Vector]](subCost: F, z: Vector, u: Vector, rho: Double)
  extends DiffFun[Vector] with Serializable {

  override def doCompute(x: Vector): (Double, Vector) = {
    val (l, g) = subCost.compute(x)
    val grad = x.copy
    BLAS.axpy(-1.0, z, grad)
    BLAS.axpy(1.0, u, grad)
    val loss = l + 0.5 * rho * BLAS.dot(grad, grad)
    BLAS.axpy(rho, grad, g)
    (loss, g)
  }

  override def doComputeInPlace(x: Vector, grad: Vector): Double = {
    throw new NotImplementedError()
  }
}

