package com.sethah.spark.sparkopt.ml

import com.sethah.spark.sparkopt.ml.optim.aggregator.DiffFunAggregator
import com.sethah.spark.sparkopt.ml.optim.loss._
import com.sethah.spark.sparkopt.ml.optim.minimizers.{HasMinimizer, IterativeMinimizer, IterativeMinimizerState}
import org.apache.spark.ml.{Estimator, InstanceWrapper, Model}
import org.apache.spark.ml.InstanceWrapper
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.param.{DoubleParam, Param, ParamMap, Params}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.sql.functions._

trait BaseAlgorithmParams extends Params with HasMinimizer {

  final val initialParams: Param[Vector] = new Param(this,
    "initialParams", "Vector of initial parameters to use as starting point in the optimizer.")

  def getInitialParams: Vector = $(initialParams)

  final val featuresCol: Param[String] = new Param(this,
    "featuresCol", "Name of column to use for feature vectors in the input dataframe.")

  def getFeaturesCol: String = $(featuresCol)

  final val labelCol: Param[String] = new Param(this, "labelCol",
    "Name of column to use for labels in the input dataframe.")

  def getLabelCol: String = $(labelCol)

  final val weightCol: Param[String] = new Param(this, "weightCol",
    "Name of column to use for sample weights in the input dataframe.")

  def getWeightCol: String = $(weightCol)

  final val l2Reg: Param[EnumeratedRegularization[Vector]] =
    new Param[EnumeratedRegularization[Vector]](this, "l2Reg", "L2 regularization instance.")

  def getL2Reg: EnumeratedRegularization[Vector] = $(l2Reg)

  final val l1Reg: Param[EnumeratedRegularization[Vector]] =
    new Param[EnumeratedRegularization[Vector]](this, "l1Reg", "L1 regularization instance")

  def getL1Reg: EnumeratedRegularization[Vector] = $(l1Reg)

}

class BaseAlgorithm(override val uid: String,
                    instanceLoss: InstanceWrapper.Instance => DiffFun[Vector])
  extends Estimator[BaseAlgorithmModel] with BaseAlgorithmParams {

  def this(instanceLoss: InstanceWrapper.Instance => DiffFun[Vector]) =
    this(Identifiable.randomUID("base"), instanceLoss)

  override type MinimizerType = IterativeMinimizer[Vector,
    SeparableDiffFun[RDD], IterativeMinimizerState[Vector]]

  override def transformSchema(schema: StructType): StructType = {
    StructType(schema.fields :+ StructField("predictions", DoubleType, false))
  }

  override def copy(extra: ParamMap): BaseAlgorithm = defaultCopy(extra)

  def setL2Reg(value: EnumeratedRegularization[Vector]): this.type = set(l2Reg, value)

  def setL1Reg(value: EnumeratedRegularization[Vector]): this.type = set(l1Reg, value)

  def setMinimizer(value: MinimizerType): this.type = set(minimizer, value)

  def setInitialParams(value: Vector): this.type = set(initialParams, value)

  def setFeaturesCol(value: String): this.type = set(featuresCol, value)
  setDefault(featuresCol -> "features")

  def setWeightCol(value: String): this.type = set(weightCol, value)
  setDefault(weightCol -> "")

  def setLabelCol(value: String): this.type = set(labelCol, value)
  setDefault(labelCol -> "label")

  override def fit(dataset: Dataset[_]): BaseAlgorithmModel = {
    val w = if ($(weightCol).isEmpty) lit(1.0) else col($(weightCol))
    val instances: RDD[InstanceWrapper.Instance] =
      dataset.select(col($(labelCol)), w, col($(featuresCol))).rdd.map {
        case Row(label: Double, weight: Double, features: Vector) =>
          InstanceWrapper.create(label, weight, features)
      }
    instances.persist(StorageLevel.MEMORY_AND_DISK)

    val losses: RDD[DiffFun[Vector]] = instances.map(instanceLoss)

    val getAggregatorFunc: Vector => DiffFunAggregator = new DiffFunAggregator(_)
    val costFun = new SeparableDiffFun[RDD](losses, getAggregatorFunc, getRegularizers(),
      cache = true)
    val (lastIter, lossHistory) = $(minimizer).takeLast(costFun, getInitialParams)
    instances.unpersist(blocking = false)
    copyValues(new BaseAlgorithmModel(uid, lastIter.params, lossHistory))
  }

  private def getRegularizers(): List[EnumeratedRegularization[Vector]] = {
    if (isSet(l2Reg) && isSet(l1Reg)) List($(l2Reg), $(l1Reg))
    else if (isSet(l1Reg)) List($(l1Reg))
    else if (isSet(l2Reg)) List($(l2Reg))
    else List.empty
  }
}

class BaseAlgorithmModel(
    override val uid: String,
    val coefficients: Vector,
    val lossHistory: Array[Double])
  extends Model[BaseAlgorithmModel] with BaseAlgorithmParams {

  override def transform(dataset: Dataset[_]): DataFrame = {
    throw new NotImplementedError("Base algorithm doesn't support transform")
  }

  override def transformSchema(schema: StructType): StructType = schema

  override def copy(extra: ParamMap): BaseAlgorithmModel = defaultCopy(extra)

}

