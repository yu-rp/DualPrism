import torch

from torch_geometric.utils import degree, to_dense_adj
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data

class Graph:
    def __init__(
        self,
        A: torch.Tensor = None,  # adjacent matrix
        Lambda: torch.Tensor = None,  # eigen values in spectral space
        U: torch.Tensor = None,  # eigen vectors in spectral space
        threshold: float = 0.5,  # used to convert to unweighted graph
    ):
        self.threshold = threshold
        if A is not None:
            assert torch.all(A >= 0), "only support positive adj matrix"
            self.A = A
            self.Lambda, self.U = self.get_spectral()
        else:
            U /= torch.sqrt(torch.sum(U**2, dim=0, keepdim=True))
            assert torch.all(torch.round(torch.matmul(U.T, U)*1000) == torch.eye(U.shape[0])*1000), "only support orthonormal eigen vectors"
            self.Lambda = Lambda
            self.U = U
            self.A = self.get_A()

    def get_A(self):
        L = torch.matmul(torch.matmul(self.U, self.Lambda), self.U.T)
        A = -L.clone()
        A[torch.eye(L.shape[0], dtype=torch.bool)] = 0
        A = torch.round(A * 1e5) / 1e5
        A = torch.where(A < self.threshold, torch.tensor(0), torch.tensor(1))
        return A

    def get_spectral(self):
        D = torch.diag(torch.sum(self.A, dim=1))
        L = D - self.A

        eigenValues, eigenVectors = torch.symeig(L, eigenvectors=True)

        idx = eigenValues.argsort(descending=True)
        eigenValues = torch.diag(eigenValues[idx])
        eigenVectors = eigenVectors[:, idx]

        return eigenValues, eigenVectors

# def add_noise(array, std_dev):
#     # Add Gaussian noise
#     noise = numpy.random.normal(0, std_dev, len(array)-1)
#     noisy_array = array[:-1] + noise

#     # Ensure all values are greater than 0
#     noisy_array = numpy.where(noisy_array > 0, noisy_array, 0.01)

#     # Sort the array from largest to smallest
#     sorted_noisy_array = numpy.sort(noisy_array)[::-1]

#     # Append 0 to the end of the array
#     final_array = numpy.append(sorted_noisy_array, 0)

#     return final_array

def add_noise(array, std_dev):
    if array.numel() == 0:
        return array
    else:
        # Add Gaussian noise
        noise = torch.randn(len(array) - 1) * std_dev
        noisy_array = array[:-1] + noise

        # Ensure all values are greater than 0
        noisy_array = torch.where(noisy_array > 0, noisy_array, torch.tensor(0.01))

        # Sort the array from largest to smallest
        sorted_noisy_array = torch.sort(noisy_array, descending=True)[0]

        # Append 0 to the end of the array
        final_array = torch.cat((sorted_noisy_array, torch.tensor([0])))

        return final_array

def generate_threshold_vector(k, threshold):
    random_vector = torch.rand(k)
    
    threshold_vector = random_vector <= threshold
    
    return threshold_vector

def _spectral_noise_(graph, **kwargs):


    adj = to_dense_adj(graph.edge_index)[0]
    g = Graph(adj)

    Lambda = g.Lambda
    U = g.U

    std_dev = kwargs["std_dev"]

    eigenvalue_indices = freq2indices(Lambda.shape[0], kwargs["aug_freq"], kwargs["aug_freq_ratio"])
    aug_prob = kwargs["aug_prob"]

    threshold_vector = generate_threshold_vector(len(eigenvalue_indices), aug_prob)
    eigenvalue_indices = eigenvalue_indices[threshold_vector]
    Lambda[eigenvalue_indices,eigenvalue_indices] = add_noise(torch.diag(Lambda)[eigenvalue_indices], std_dev)

    g = Graph(
        Lambda = Lambda,
        U = U
    )

    edge_index, _ = dense_to_sparse(g.A)

    num_nodes = int(torch.max(edge_index)) + 1

    pyg_graph = Data()
    pyg_graph.y = graph.y
    pyg_graph.x = graph.x
    pyg_graph.edge_index = edge_index
    pyg_graph.num_nodes = num_nodes

    return pyg_graph

def spectral_noise(dataset, **kwargs):

    import pdb
    pdb.set_trace()

    augment_count = int(dataset * kwargs["aug_ratio"])
    indices = np.random.choice(len(dataset), size=augment_count, replace=False)

    for index in indices:
        graph = dataset[index]
        dataset[index] = _spectral_noise_(graph, **kwargs)

    return dataset

def _spectral_mask_(graph, **kwargs):

    adj = to_dense_adj(graph.edge_index)[0]
    g = Graph(adj)

    Lambda = g.Lambda
    U = g.U

    eigenvalue_indices = freq2indices(Lambda.shape[0], kwargs["aug_freq"], kwargs["aug_freq_ratio"])
    aug_prob = kwargs["aug_prob"]

    threshold_vector = generate_threshold_vector(len(eigenvalue_indices), aug_prob)
    eigenvalue_indices = eigenvalue_indices[threshold_vector]
    Lambda[eigenvalue_indices,eigenvalue_indices] *= 0.0 

    g = Graph(
        Lambda = Lambda,
        U = U
    )

    edge_index, _ = dense_to_sparse(g.A)

    num_nodes = int(torch.max(edge_index)) + 1

    pyg_graph = Data()
    pyg_graph.y = graph.y
    pyg_graph.x = graph.x
    pyg_graph.edge_index = edge_index
    pyg_graph.num_nodes = num_nodes

    return pyg_graph

def spectral_mask(dataset, **kwargs):

    sample_graphs = []

    for index in kwargs["aug_indices"]:
        graph = dataset[index]
        sample_graphs.append(_spectral_mask_(graph, **kwargs))

    return sample_graphs

def freq2indices(N, freq, ratio):
    indices = torch.arange(N)
    if freq is None:
        pass
    elif freq == "low":
        count = int(N * ratio)
        indices = indices[:count]
    elif freq == "high":
        count = int(N * ratio)
        indices = indices[-count:]
    else:
        raise RuntimeError
    return indices