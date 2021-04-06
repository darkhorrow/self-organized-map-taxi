
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np

class Question(ABC):

    @staticmethod
    def _plot_tags(tags, locs, x_size, y_size, clases=''):
        # Seleccion de colores por clase
        if clases == '':
            clases = list(set(tags))

        cmap = plt.get_cmap('viridis')
        col = cmap(np.linspace(0, 1, len(clases)))
        colores = {}
        for i, c in enumerate(clases):
            colores[c] = col[i]

        # Representacion del mapa
        fig = plt.figure(1)
        ax = fig.add_subplot(111)

        mapa = np.zeros((y_size + 1, x_size + 1, 4), dtype=np.float32)
        mapa[:, :, 3] = 1
        for tag, (i, j) in zip(tags, locs):
            if np.array_equal(mapa[i, j, :], [0, 0, 0, 1]):
                mapa[i, j, :] = colores[tag]
            else:
                mapa[i, j, :] += colores[tag]
                mapa[i, j, :] = np.divide(mapa[i, j, :], 2)

        # Leyenda
        cuadro_color, etiqueta = [], []
        for name_color, value_color in colores.items():
            rect = plt.Rectangle((0, 0), 1, 1, fc=value_color)
            cuadro_color.append(rect)
            etiqueta.append(name_color)

        ax.imshow(mapa)

        ax.legend(tuple(cuadro_color), tuple(etiqueta), loc='center right', bbox_to_anchor=(1.3, 0.5))

        # Ajuste de ejes
        plt.xticks(np.arange(0, x_size + 1, np.floor_divide(x_size + 1, 5), dtype=np.int))
        plt.yticks(np.arange(0, y_size + 1, np.floor_divide(y_size + 1, 5), dtype=np.int))
        plt.xlim(0.5, x_size - 0.5)
        plt.ylim(0.5, y_size - 0.5)

        plt.show()

    @abstractmethod
    def _prepare_question(self):
        pass

    @abstractmethod
    def _question(self, px, tags, clases=''):
        pass