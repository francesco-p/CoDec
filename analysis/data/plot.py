import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#import sys
#sys.path.append("../../")
#import conf

class CSV_Plot:
    def __init__(self,ns, densities, refinement_type, CSV_PATH, SAVE_PATH, metrics):
        self.ns = ns
        self.densities = densities
        self.refinement_type = refinement_type
        self.CSV_PATH = CSV_PATH
        self.SAVE_PATH = SAVE_PATH
        self.metrics = metrics

    def plot(self):
        for n in self.ns:
            csv_file = self.CSV_PATH + f'{self.refinement_type}/{n}.csv'
            print(csv_file)

            for metric in self.metrics:
                df = pd.read_csv(csv_file, delimiter=',', index_col=1)
                ref = df['refinement'].values[0]

                data_to_plot = []
                for d in self.densities:
                    d = round(d, 2)
                    data_to_plot.append(df.loc[d, metric].values)

                # Generate figure
                fig = plt.figure(figsize=(9, 6))
                plt.ylabel(metric)
                plt.xlabel('Density')
                plt.title(f"N = {n} ref={ref}")
                plt.grid()
                ax = fig.add_subplot(111)
                ax.boxplot(data_to_plot)
                ax.set_xticklabels(self.densities)

                figname = self.SAVE_PATH + f'{metric}/' + f'{n}_{metric}_{ref}.png'
                plt.savefig(figname)


if __name__ == '__main__':
    ns = [200, 400, 800, 1000, 2000]
    densities = [0.3, 0.35, 0.40, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.80, 0.85, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1]
    SAVE_PATH = './plots/'
    refinement_type = 'indeg_guided'
    CSV_PATH = './csv/'
    metrics=['L2_distance','sze_idx','kld_1','kld_2']
    p = CSV_Plot(ns, densities[::-1], refinement_type, CSV_PATH, SAVE_PATH, metrics)
    p.plot()
    p.refinement_type = 'degree_based'
    p.plot()

    #montage *.png -tile 2x0 -geometry 1000 assemply.pdf
