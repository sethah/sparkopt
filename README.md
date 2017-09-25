This project provides flexible abstractions for building machine learning
models/algorithms using Apache Spark ML. Specifically, the following are
easily customizable:

* prediction function
* loss function
* optimization routine

Being able to easily plug in the custom components above allows users to
improve the scale of their algorithms, express a richer set of algorithms
than what is currently available in Spark ML, and even improve upon
existing Spark ML algorithms.

## Build

### Build Spark-2.3.0-SNAPSHOT

````
git clone https://github.com/apache/spark
cd spark
build/mvn clean install -DskipTests -Dmaven.javadoc.skip=True
````

````
cd [this repo]
mvn package
````

## Run example

````
$SPARK_HOME/bin/spark-submit \
--class com.sethah.spark.sparkopt.examples.LogisticRegressionExample \
target/sparkopt-1.0-SNAPSHOT-jar-with-dependencies.jar \
--trainPath src/main/resources/binary \
--minimizer admm \
--l1Reg 0.05 \
--leReg 0.05
````