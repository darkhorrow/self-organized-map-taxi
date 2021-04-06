from question import Question
from som import Som

"""
¿Influye el lugar de llegada y la distancia recorrida con la tarifa?
"""
class Question_3(Question):

    def __init__(self, dc, niters):
        self.dc = dc
        self.niters = niters
        px, tags = self._prepare_question()
        self._question(px, tags)

    def _prepare_question(self):
        dataset = self.dc.full_dataset

        std_rate = dataset[dataset['ratecode_id'] == 1].sample(n=1_000)
        jfk_rate = dataset[dataset['ratecode_id'] == 2].sample(n=1_000)
        newark_rate = dataset[dataset['ratecode_id'] == 3]
        nassau_rate = dataset[dataset['ratecode_id'] == 4]
        negotiated_rate = dataset[dataset['ratecode_id'] == 5].sample(n=1_000)
        group_rate = dataset[dataset['ratecode_id'] == 6]

        dataset = std_rate
        dataset = dataset.append(jfk_rate, ignore_index=True)
        dataset = dataset.append(newark_rate, ignore_index=True)
        dataset = dataset.append(nassau_rate, ignore_index=True)
        dataset = dataset.append(negotiated_rate, ignore_index=True)
        dataset = dataset.append(group_rate, ignore_index=True)

        # Cambiamos etiquetas 0,1 por Y,N
        dataset['ratecode_id'] = dataset['ratecode_id'].replace(1, 'Standard rate')
        dataset['ratecode_id'] = dataset['ratecode_id'].replace(2, 'JFK')
        dataset['ratecode_id'] = dataset['ratecode_id'].replace(3, 'Newark')
        dataset['ratecode_id'] = dataset['ratecode_id'].replace(4, 'Nassau or Westchester')
        dataset['ratecode_id'] = dataset['ratecode_id'].replace(5, 'Negotiated fare')
        dataset['ratecode_id'] = dataset['ratecode_id'].replace(6, 'Group ride')

        df = self.dc.normalize(dataset[['dropoff_longitude', 'dropoff_latitude', 'trip_distance', 'ratecode_id']],
                               exclude_columns=['ratecode_id'])

        df = df.sample(frac=1)
        tags = df['ratecode_id'].to_numpy()
        px = df[['dropoff_longitude', 'dropoff_latitude', 'trip_distance']].to_numpy()

        return px, tags

    def _question(self, px, tags, clases=''):
        x_size, y_size = 30, 30
        som = Som(number_inputs_each_neuron=px.shape[1], n_neurons_x=x_size, n_neurons_y=y_size, lr0=0.1)
        som.train(px, tags, number_iterations=self.niters)

        locs = []
        for tag, x in zip(tags, px):
            i, j = som.classify(x)
            locs.append((i, j))

        self._plot_tags(tags, locs, x_size, y_size)