import sys
sys.path.append("../../../../../dlapplication-dev")
sys.path.append("../../../../../dlplatform-dev")


from environments.local_environment import Experiment
from environments.datasources import FileDataSourceFactory
from environments.datasources.dataDecoders.kerasDataDecoders import MNISTDecoder
from utilss.models.keras.MNISTNetwork import MNISTCNNNetwork
from DLplatform.synchronizing import DynamicSync
from DLplatform.aggregating import Average,GeometricMedian
from DLplatform.learning.factories.kerasLearnerFactory import KerasLearnerFactory
from DLplatform.stopping import MaxAmountExamples
from DLplatform.coordinator import InitializationHandler

if __name__ == "__main__":
    executionMode = 'cpu'
    messengerHost = 'localhost'
    messengerPort = 5672
    numberOfNodes = 2
    updateRule = "sgd"
    learningRate = 0.01
    lossFunction = "categorical_crossentropy"
    batchSize = 1
    delta = 0.1
    syncPeriod = 1

    sync = DynamicSync(delta)
    aggregator = Average()
    #aggregator = GeometricMedian()
    stoppingCriterion = MaxAmountExamples(2800) #2800
    dsFactory = FileDataSourceFactory(filename = "../../../../data/cnn_mnist/noisymnist.csv", decoder = MNISTDecoder(), numberOfNodes = numberOfNodes, indices = 'roundRobin', shuffle = False, cache = False)
    learnerFactory = KerasLearnerFactory(network=MNISTCNNNetwork(), updateRule=updateRule, learningRate=learningRate, lossFunction=lossFunction, batchSize=batchSize, syncPeriod=syncPeriod, delta=delta, aggType='average') #geometeric_median
    initHandler = InitializationHandler()

    exp = Experiment(executionMode = executionMode, messengerHost = messengerHost, messengerPort = messengerPort,
        numberOfNodes = numberOfNodes, sync = sync,
        aggregator = aggregator, learnerFactory = learnerFactory,
        dataSourceFactory = dsFactory, stoppingCriterion = stoppingCriterion, initHandler = initHandler)
    exp.run("MNISTkerasCNN")
