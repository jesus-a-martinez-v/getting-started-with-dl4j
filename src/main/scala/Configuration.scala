import com.typesafe.config.ConfigFactory

object Configuration {
  private val configuration = ConfigFactory.load()
  private val parametersConfig = configuration.getConfig("parameters")
  private val hyperParametersConfig = configuration.getConfig("hyperParameters")

  val numberOfRows: Int = parametersConfig.getInt("numberOfRows")
  val numberOfColumns: Int = parametersConfig.getInt("numberOfColumns")
  val numberOfOutputClasses: Int = parametersConfig.getInt("numberOfOutputClasses")

  val batchSize: Int = hyperParametersConfig.getInt("batchSize")
  val randomSeed: Int = hyperParametersConfig.getInt("randomSeed")
  val numberOfEpochs: Int = hyperParametersConfig.getInt("numberOfEpochs")
  val regularizationRate: Double = hyperParametersConfig.getDouble("regularizationRate")
}
