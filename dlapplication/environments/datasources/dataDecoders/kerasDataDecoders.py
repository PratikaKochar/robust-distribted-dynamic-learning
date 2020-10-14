from environments.datasources.dataDecoders import DataDecoder, CSVDecoder
import numpy as np

class MNISTDecoder(CSVDecoder):    
    def __init__(self, delimiter = ',', labelCol = 0):
        CSVDecoder.__init__(self, delimiter, labelCol)
        
    def __call__(self, line):
        parsed_line = [float(c) for c in line.split(self._delimiter)]
        image = np.asarray(parsed_line[1:], dtype='float32').reshape(28,28,1) / 255.0
        label = np.zeros(10)
        label[int(parsed_line[self._labelCol])] = 1
            
        return image, label

    def __str__(self):
        return "MNIST text file for keras"
    
class CifarDecoder(CSVDecoder):
    def __init__(self, numClasses = 10):
        self._numClasses = numClasses
        
    def __call__(self, line):
        parsed_line = []
        d = '.'
        for c in line.split(','):
            counter = c.count(d)
            if counter > 1:
                b = d.join(c.split(d, 2)[:2])
                c = b
            parsed_line.append(float(c))
        image = np.asarray(parsed_line[1:], dtype='float32').reshape(32, 32, 1) #assumin data is normalised
        label = np.zeros(self._numClasses)
        label[int(parsed_line[0])] = 1

        return image, label
    
    def __str__(self):
        return "Cifar10 or 100 from text file for keras"
        
class TrajectoryDecoder(DataDecoder):   
    def __call__(self, line):
        parsed_traj = [float(c) for c in line.split(',')]
        return parsed_traj, 0

    def __str__(self):
        return "Trajectories of pedestrians"



