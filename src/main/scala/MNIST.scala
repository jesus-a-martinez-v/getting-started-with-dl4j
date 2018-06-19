import org.deeplearning4j.datasets.iterator.BaseDatasetIterator
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

object MNIST extends App {
  // Data
  val trainingSet = getTrainData(Configuration.batchSize, Configuration.randomSeed)
  val testSet = getTestData(Configuration.batchSize, Configuration.randomSeed)

  val networkConfiguration = getNerworkConfiguration
  // Actual model
  val network = new MultiLayerNetwork(networkConfiguration)

  network.init()
  network.setListeners(new ScoreIterationListener(1))  // Prints scores each iteration

  // Train network and then evaluate performance
  train(network, trainingSet, Configuration.numberOfEpochs)
  evaluate(network, testSet, Configuration.numberOfOutputClasses)

  private def getTrainData(batchSize: Int, seed: Int = 42) = new MnistDataSetIterator(batchSize, true, seed)

  private def getTestData(batchSize: Int, seed: Int = 42) = new MnistDataSetIterator(batchSize, false, seed)

  private def getNerworkConfiguration: MultiLayerConfiguration = {
    // Network layers
    val firstLayer = getHiddenLayer(Configuration.numberOfColumns * Configuration.numberOfRows, 1000)
    val outputLayer = getOutputLayer(1000, Configuration.numberOfOutputClasses)

    // Network configuration
    new NeuralNetConfiguration.Builder()
      .seed(Configuration.randomSeed)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .iterations(1)
      .updater(Updater.ADAM)
      .regularization(true)
      .l2(Configuration.regularizationRate)
      .list(firstLayer, outputLayer)
      .pretrain(false)
      .backprop(true)
      .build()
  }
  private def getHiddenLayer(inputSize: Int,
                             outputSize: Int,
                             activation: Activation = Activation.RELU,
                             weightInitializationMethod: WeightInit = WeightInit.XAVIER) =
    new DenseLayer.Builder()
      .nIn(inputSize)
      .nOut(outputSize)
      .activation(activation)
      .weightInit(weightInitializationMethod)
      .build()

  private def getOutputLayer(inputSize: Int,
                             outputSize: Int,
                             activation: Activation = Activation.SOFTMAX,
                             weightInitializationMethod: WeightInit = WeightInit.XAVIER,
                             lossFunction: LossFunction = LossFunction.NEGATIVELOGLIKELIHOOD) =
    new OutputLayer.Builder(lossFunction)
      .nIn(inputSize)
      .nOut(outputSize)
      .activation(activation)
      .weightInit(weightInitializationMethod)
      .build()

  private def train(model: MultiLayerNetwork, data: BaseDatasetIterator, epochs: Int): Unit =
    for (epoch <- 1 to epochs ) {
      println(s"Epoch $epoch")
      model fit data
    }

  private def evaluate(model: MultiLayerNetwork, testData: BaseDatasetIterator, outputClasses: Int): Unit = {
    val evaluation = new Evaluation(outputClasses)

    while (testData.hasNext) {
      val sample = testData.next()
      val output = model.output(sample.getFeatureMatrix)

      evaluation.eval(sample.getLabels, output)
    }

    println(evaluation.stats())
  }
}