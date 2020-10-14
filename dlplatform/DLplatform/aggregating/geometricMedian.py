from DLplatform.aggregating import Aggregator

from DLplatform.parameters import Parameters
from typing import List
import hdmedians as hd
import numpy as np

class GeometricMedian(Aggregator):
    '''
    Provides a method to calculate an aggregated model from n individual models (using the geometric median)
    '''

    def __init__(self, name="GeometricMedian"):
        '''

        Returns
        -------
        None
        '''
        Aggregator.__init__(self, name=name)

    def __call__(self, params: List[Parameters]) -> Parameters:
        '''

        This aggregator takes n lists of model parameters and returns a list of component-wise geometric median.

        Parameters
        ----------
        params A list of Paramters objects. These objects support addition and scalar multiplication.

        Returns
        -------
        A new parameter object that is the geometric median of params.

        '''

        models = params

        shapes = []
        b = []
        once = True
        newWeightsList = []
        try:
            for i, model in enumerate(models):
                w2 = model.get()
                c = []
                c = np.array(c)
                for i in range(len(w2)):
                    z = np.array(w2[i])

                    if len(shapes) < len(w2):
                        shapes.append(z.shape)
                    d = np.array(w2[i].flatten()).squeeze()
                    c = np.concatenate([c, d])
                if (once):
                    b = np.zeros_like(c)
                    b[:] = c[:]
                    once = False
                else:
                    once = False
            b = np.concatenate([b.reshape((-1, 1)), c.reshape((-1, 1))], axis=1)
            median_val = np.array(hd.geomedian(b))
            sizes = []
            for j in shapes:
                size = 1
                for k in j:
                    size *= k
                sizes.append(size)
            newWeightsList = []

            chunks = []
            count = 0
            for size in sizes:
                chunks.append([median_val[i + count] for i in range(size)])
                count += size
            for chunk, i in zip(chunks, range(len(shapes))):
                newWeightsList.append(np.array(chunk).reshape(shapes[i]))

        except Exception as e:
            print("Error happened! Message is ", e)
        newParams = params[0].getCopy()
        return newParams.set(newWeightsList)

    def __str__(self):
        return "Geometric Median"
