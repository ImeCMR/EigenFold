from .sde import HarmonicSDE
import numpy as np

class PolymerSDE(HarmonicSDE):
    def __init__(self, N, a, b, args=None):
        self.args = args
        if self.args is not None and self.args.dataset_type == 'noesy':
            # Edges will be provided in the dataset
            super(PolymerSDE, self).__init__(N=N, a=a, b=b, args=args)
        else:
            super(PolymerSDE, self).__init__(N=N,
                edges=zip(np.arange(N-1), np.arange(1, N)),
                antiedges=zip(np.arange(N-2), np.arange(2, N)),
            a=a, b=b, args=args)

    def add_noise_to_data(self, data):
        if self.args is not None and self.args.dataset_type == 'noesy':
            edge_info = []
            for i in range(data['resi'].edge_index.shape[1]):
                src = data['resi'].edge_index[0, i].item()
                dst = data['resi'].edge_index[1, i].item()
                dist = data['resi'].edge_attr[i, 0].item()
                peak_type = data['resi'].edge_attr[i, 1].item()
                edge_info.append((src, dst, dist, peak_type))

            self.__init__(N=data['resi'].num_nodes, a=self.args.sde_a, b=self.args.sde_b, args=self.args)
            self.D, self.P = self.D.copy(), self.P.copy() # Avoid issues with caching
            self.edges = edge_info
        return data
    