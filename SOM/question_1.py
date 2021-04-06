from data_cleaner import DataCleaner
from question import Question
from som import Som

import pandas as pd

'''
¿Existe alguna relación entre el coste total y la distancia, y tipo de tarifa?
'''
class Question_1(Question):

    def __init__(self, dc, niters=10):
        self.dc = dc
        self.niters = niters
        px, tags, clases = self._prepare_question()
        self._question(px, tags, clases)

    def _prepare_question(self):

        df = self.dc.full_dataset[['trip_distance', 'ratecode_id', 'total_amount']]

        rango1 = df.query('total_amount <= 5').sample(n=1_000)
        rango2 = df.query('total_amount > 5 and total_amount <= 10').sample(n=1_000)
        rango3 = df.query('total_amount > 10 and total_amount <= 15').sample(n=1_000)
        rango4 = df.query('total_amount > 15 and total_amount <= 25').sample(n=1_000)
        rango5 = df.query('total_amount > 25 and total_amount <= 35').sample(n=1_000)
        rango6 = df.query('total_amount > 35').sample(n=1_000)

        rango1['total_amount'] = 0
        rango2['total_amount'] = 1
        rango3['total_amount'] = 2
        rango4['total_amount'] = 3
        rango5['total_amount'] = 4
        rango6['total_amount'] = 5

        df = rango1
        df = df.append(rango2, ignore_index=True)
        df = df.append(rango3, ignore_index=True)
        df = df.append(rango4, ignore_index=True)
        df = df.append(rango5, ignore_index=True)
        df = df.append(rango6, ignore_index=True)

        df = df.sample(frac=1)

        amount = df[['total_amount']]
        values = df[['trip_distance', 'ratecode_id']]

        values = pd.concat([values, pd.get_dummies(values['ratecode_id'], prefix='ratecode_id')], axis=1)
        values.drop(['ratecode_id'], axis=1, inplace=True)

        values = self.dc.normalize(values)

        clases = []
        clases.append("[0-5]")
        clases.append("[5-10]")
        clases.append("[10-15]")
        clases.append("[15-25]")
        clases.append("[25-35]")
        clases.append("[+35]")

        for i in range(len(clases)):
            amount = amount.replace(i, clases[i], regex=True)

        px = values.to_numpy()
        tags = amount.to_numpy()[:, 0]

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
