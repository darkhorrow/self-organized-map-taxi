
from question import Question
from som import Som

import pandas as pd

'''
¿Existe alguna relación entre la propina y el número de pasajeros, la distancia del recorrido y el método de pago?
'''
class Question_2(Question):

    def __init__(self, dc, niters=10):
        self.dc = dc
        self.niters = niters
        px, tags, clases = self._prepare_question()
        self._question(px, tags, clases)

    def _prepare_question(self):

        df = self.dc.full_dataset[['trip_distance', 'passenger_count', 'payment_type', 'tip_amount']]

        rango1 = df.query('tip_amount <= 0.5').sample(n=1_000)
        rango2 = df.query('tip_amount > 0.5 and tip_amount <= 1').sample(n=1_000)
        rango3 = df.query('tip_amount > 1 and tip_amount <= 1.5').sample(n=1_000)
        rango4 = df.query('tip_amount > 1.5 and tip_amount <= 3').sample(n=1_000)
        rango5 = df.query('tip_amount > 3 and tip_amount <= 6').sample(n=1_000)
        rango6 = df.query('tip_amount > 6').sample(n=1_000)

        rango1['tip_amount'] = 0
        rango2['tip_amount'] = 1
        rango3['tip_amount'] = 2
        rango4['tip_amount'] = 3
        rango5['tip_amount'] = 4
        rango6['tip_amount'] = 5

        df = rango1
        df = df.append(rango2, ignore_index=True)
        df = df.append(rango3, ignore_index=True)
        df = df.append(rango4, ignore_index=True)
        df = df.append(rango5, ignore_index=True)
        df = df.append(rango6, ignore_index=True)

        df = df.sample(frac=1)

        tip = df[['tip_amount']]
        values = df[['trip_distance', 'passenger_count', 'payment_type']]

        values = pd.concat([values, pd.get_dummies(values['payment_type'], prefix='payment_type')], axis=1)
        values.drop(['payment_type'], axis=1, inplace=True)

        values = self.dc.normalize(values)

        clases = []
        clases.append("[0-0.5]")
        clases.append("[0.5-1]")
        clases.append("[1-1.5]")
        clases.append("[1.5-3]")
        clases.append("[3-6]")
        clases.append("[+6]")

        for i in range(len(clases)):
            tip = tip.replace(i, clases[i], regex=True)

        px = values.to_numpy()
        tags = tip.to_numpy()[:, 0]

        return px, tags, clases

    def _question(self, px, tags, clases=''):
        x_size, y_size = 20, 20
        som = Som(number_inputs_each_neuron=px.shape[1], n_neurons_x=x_size, n_neurons_y=y_size, lr0=0.1)
        som.train(px, tags, number_iterations=self.niters)

        locs = []
        for tag, x in zip(tags, px):
            i, j = som.classify(x)
            locs.append((i, j))

        self._plot_tags(tags, locs, x_size, y_size, clases)