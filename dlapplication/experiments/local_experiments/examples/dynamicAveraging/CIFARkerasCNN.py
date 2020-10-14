import sys
sys.path.append("../../../../../dlapplication-dev")
sys.path.append("../../../../../dlplatform-dev")


from environments.local_environment import Experiment
from environments.datasources import FileDataSourceFactory
from environments.datasources.dataDecoders.kerasDataDecoders import CifarDecoder
from utilss.models.keras.CIFAR100Network import CIFAR100Network
from DLplatform.synchronizing import DynamicSync
from DLplatform.aggregating import Average,GeometricMedian
from DLplatform.learning.factories.kerasLearnerFactory import KerasLearnerFactory
from DLplatform.stopping import MaxAmountExamples
from DLplatform.coordinator import InitializationHandler

if __name__ == "__main__":
    executionMode = 'cpu'
    messengerHost = 'localhost'
    messengerPort = 5672
    numberOfNodes = 3
    updateRule = "sgd"
    learningRate = 0.01
    lossFunction = "categorical_crossentropy"
    batchSize = 1
    delta = 0.1
    syncPeriod = 1

    sync = DynamicSync(delta)
    aggregator = GeometricMedian()
    stoppingCriterion = MaxAmountExamples(100) #2800
    dsFactory = FileDataSourceFactory(filename = "../../../../data/cifar/noisycifar100.csv", decoder = CifarDecoder(), numberOfNodes = numberOfNodes, indices = 'roundRobin', shuffle = False, cache = False)
    learnerFactory = KerasLearnerFactory(network=CIFAR100Network(), updateRule=updateRule, learningRate=learningRate, lossFunction=lossFunction, batchSize=batchSize, syncPeriod=syncPeriod, delta=delta, aggType='geometeric_median')
    initHandler = InitializationHandler()

    exp = Experiment(executionMode = executionMode, messengerHost = messengerHost, messengerPort = messengerPort,
        numberOfNodes = numberOfNodes, sync = sync,
        aggregator = aggregator, learnerFactory = learnerFactory,
        dataSourceFactory = dsFactory, stoppingCriterion = stoppingCriterion, initHandler = initHandler)
    exp.run("CIFARkerasCNN")
