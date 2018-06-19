name := "GettingStartedWithDL4J"

version := "0.1"

scalaVersion := "2.12.4"

classpathTypes += "maven-plugin"

// https://mvnrepository.com/artifact/org.deeplearning4j/deeplearning4j-core
libraryDependencies ++= Seq(
  "org.nd4j" % "nd4j-native-platform" % "0.9.1",
  "org.deeplearning4j" % "deeplearning4j-core" % "0.9.1",
  "com.typesafe" % "config" % "1.3.3"
)