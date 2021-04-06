
import tensorflow as tf
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Som:

    def __init__(self, number_inputs_each_neuron=2, n_neurons_x=2, n_neurons_y=2, lr0=0.1):
        self._graph = tf.Graph()
        self._number_inputs_each_neuron = number_inputs_each_neuron
        self._n_neurons_x = n_neurons_x
        self._n_neurons_y = n_neurons_y

        with self._graph.as_default():
            self._initial_radius = tf.cast(tf.maximum(self._n_neurons_x, self._n_neurons_y), tf.float32)/2
            self._lr0 = tf.Variable(lr0, dtype=tf.float32)
            self.w = tf.Variable(tf.random.normal([n_neurons_x*n_neurons_y, number_inputs_each_neuron]))

    def neurons_indexes(self):
        """
        [[0,0],
         [0,1],
         [0,2],
          ...
         [0,x],
          ...
         [y,0],
          ...
         [y,x]]
        :return: array with position
        """
        indexes = []
        for i in range(self._n_neurons_y):
            for j in range(self._n_neurons_x):
                indexes.append([i,j])
        return indexes

    def _get_bmu(self, x):
        """
        :param x:
        :return: argmin(w[i,:] - x) i=1..n_neurons
        """
        with self._graph.as_default():
            # Replicar la entrada al número de neuronas para calcular distancia
            n_neurons = tf.constant([self._n_neurons_x*self._n_neurons_y])
            superx = tf.reshape(tf.tile(x, n_neurons), [n_neurons[0], tf.shape(x)[0]])

            # Calcular la diferencia minima
            diff = tf.math.squared_difference(self.w, superx)
            diff = tf.math.reduce_sum(diff, axis=1)
            idx = tf.math.argmin(diff, axis=0)

            # Calcular indice en la matriz de neuronas
            imin = tf.mod(idx, self._n_neurons_x)
            jmin = tf.math.floordiv(idx, self._n_neurons_x)
            return tf.stack([tf.cast(imin, tf.float32), tf.cast(jmin, tf.float32)])

    def _get_distance(self, bmu_loc):
        # Replicar posicion para calcular distancia
        n_neurons = tf.constant([self._n_neurons_x * self._n_neurons_y])
        BMU = tf.reshape(tf.tile(bmu_loc, n_neurons), [n_neurons[0], tf.shape(bmu_loc)[0]])
        locations = tf.Variable(self.neurons_indexes(), dtype=tf.float32)
        dist = tf.math.squared_difference(BMU, locations)
        dist = tf.math.reduce_sum(dist, axis=1)
        return dist

    def _get_distance_torus(self, bmu_loc):
        """
        Dt_ij(bmu_mn, n_ij) = min{|m-i|, y-|m-i|}**2 + min{|j-k|, x-|j-k|}**2
        Elevamos al cuadrado para evitar distancias negativas
        :param bmu_loc:
        :return: Dt
        """
        n_neurons = tf.constant([self._n_neurons_x * self._n_neurons_y])
        BMU = tf.reshape(tf.tile(bmu_loc, n_neurons), [n_neurons[0], tf.shape(bmu_loc)[0]])
        locations = tf.Variable(self.neurons_indexes(), dtype=tf.float32)

        fc = tf.Variable([[self._n_neurons_y, self._n_neurons_x] for _ in range(self._n_neurons_x*self._n_neurons_y)], dtype=tf.float32)

        diff_abs = tf.abs(tf.subtract(BMU, locations))
        diff_rc = tf.subtract(fc, diff_abs)

        DTrow = tf.concat([tf.transpose([diff_abs[:, 0]]), tf.transpose([diff_rc[:, 0]])], axis=1)
        DTcol = tf.concat([tf.transpose([diff_abs[:, 1]]), tf.transpose([diff_rc[:, 1]])], axis=1)
        DTrow = tf.transpose([tf.reduce_min(DTrow, axis=1)])
        DTcol = tf.transpose([tf.reduce_min(DTcol, axis=1)])

        DT = tf.concat([DTrow, DTcol], axis=1)

        DT = tf.square(DT)
        DT = tf.reduce_sum(DT, axis=1)
        return DT

    def _neighbours_function(self, bmu_loc, iteration, lmbd):
        """
        dist[i] = sum(bmu_loc - w[i, :])
        factor[i] = e^-dist[i]
        :param bmu: BEST MATCHING UNIT location
        :return: factor vecino
        """
        with self._graph.as_default():
            # dist = self._get_distance(bmu_loc)
            dist = self._get_distance_torus(bmu_loc)

            r = self._initial_radius*tf.exp(-1*tf.divide(iteration, lmbd))

            r_inv = tf.divide(tf.Variable(1, dtype=tf.float32), r)
            return tf.math.exp(tf.scalar_mul(0.5*r_inv[0]*r_inv[0], tf.math.scalar_mul(-1, dist)))

    def _fit(self, x_input, iteration):
        """
        i = 1..n_neurons
        dw[i,:] = lr*factor(i)*(x_input-w[i,:])
        w[i,:] <- w[i,:] + dw[i,:]
        """
        with self._graph.as_default():
            bmu = self._get_bmu(x_input)
            # Actualizar pesos de los mas cercanos

            # Lambda
            factor = self._neighbours_function(bmu, iteration, self.lmbd)

            # Learning rate adaptativo
            lr = self._lr0 * tf.exp(-1 * tf.divide(iteration, self.lmbd))

            dw = tf.math.scalar_mul(lr[0], tf.math.subtract(x_input, self.w))

            factor = tf.reshape(tf.tile(factor, [self._number_inputs_each_neuron]), [self._number_inputs_each_neuron, factor.shape[0]])
            factor = tf.transpose(factor)
            dw = tf.math.multiply(factor, dw)
            return tf.compat.v1.assign(self.w, tf.math.add(self.w, dw))

    def train(self, px_train, tags, number_iterations=100):

        with self._graph.as_default():
            self.lmbd = tf.divide(number_iterations, tf.math.log(self._initial_radius)) # lambda = iterations/ln(r0)

            self.iteration = tf.compat.v1.placeholder(dtype=tf.float32, shape=(1,), name="iteration")
            self.input = tf.compat.v1.placeholder(dtype=tf.float32, shape=self._number_inputs_each_neuron, name="input")

            self.w_update = self._fit(self.input, self.iteration)

            init = tf.compat.v1.global_variables_initializer()

        with tf.compat.v1.Session(graph=self._graph) as sess:
            sess.run(init)
            for it in range(number_iterations):
                for x in px_train:
                    sess.run(self.w_update, feed_dict={self.input: x, self.iteration: np.array([it])})
                print("\rIteration " + str(it+1) + "/" + str(number_iterations), end='')

            self.trained_w = sess.run(self.w)

    def classify(self, x):
        n_neurons = self._n_neurons_x * self._n_neurons_y
        bigX = np.reshape(np.tile(x, n_neurons), (n_neurons, self._number_inputs_each_neuron))
        distances = np.sum((self.trained_w - bigX) ** 2, axis=1)
        idx = np.argmin(distances)
        jbmu = idx % self._n_neurons_x
        ibmu = int(idx / self._n_neurons_x)
        return ibmu, jbmu


"""
    def evaluate_precision(self, px):
        e = np.zeros(px.shape[1])
        for x in px:
            i, j = self.classify(x)
            w = self.trained_w[i * self._n_neurons_x + j]
            e += abs(x-w)
        return 1-(sum(e)/(px.shape[1]*len(px)))
    
    def topological_evaluation(self, px):
        # sum[u]/N  \  u = dist[ bmu_1, bmu_2 ]
        pass

"""