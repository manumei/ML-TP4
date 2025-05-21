import cupy as cp
import numpy as np
import time
from tqdm import trange
from tqdm import tqdm
from colorama import Fore, Style
import matplotlib.pyplot as plt

# para plotear las gaussianas
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

# Aux
def draw_ellipse(mean, cov, ax, color):
    ''' matplotlib magic para dibujar una elipse aparentemente '''
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * np.sqrt(vals)
    
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=theta,
                    edgecolor=color, fc='none', lw=2, zorder=3)

    ax.add_patch(ellipse)

def has_converged(new, old, rtol, atol):
    return cp.allclose(new, old, rtol=rtol, atol=atol)

def compute_L(X, labels, centroids):
    ''' 
    Suma de distancias (L) de cada punto a su centroide (distancia euclidiana) 
    todos los inputs son cp arrays, X es el dataset y centroids son los centroides 
    cupy magic cuando haces ventroids[labels], devuelve un ndarray donde cada fila era el valor de centroids, que matcheaba el indice de labels que se le dio
    es medio raro explicarlo asi en el aire pero despues visualmente se ve bien, es solo una forma hiper eficiente de decir:
    "esta es la lista (array) de clusters a los que pertenece cada dato, devolveme la lista (array) de centroides que matchean cada uno de los datos"
    entonces, uso labels para slicear centroides y me quedo con la lista de todos los centroides en el orden 'correcto', aka el mismo orden que los datos
    y ahora que los tengo en el mismo orden, puedo hacer directamente la distancia euclidiana, me devuelve un array con todas las distancias, y cp.sum las suma
    devuelve el float de la suma nada mas
    '''
    # mucho texto, numpy magic (o cupy en este caso)
    return cp.sum(cp.linalg.norm(X - centroids[labels], axis=1)).item()

def compute_GM_Loss(X, means, covs, weights):
    '''Compute the negative log-likelihood loss of the Gaussian Mixture Model'''
    N = X.shape[0]
    K = means.shape[0]

    total_log_prob = cp.zeros(N)

    for k in range(K):
        prob = weights[k] * gaussian_pdf(X, means[k], covs[k])
        total_log_prob += prob

    log_likelihood = cp.sum(cp.log(total_log_prob + 1e-12))  # avoid log(0)
    return -log_likelihood

# camins
def kmeans(X, K, max_iters, rel_tol, abs_tol):
    """K-means clustering on dataset (cupy for GPU)

    Args:
        X (cp.ndarray): Dataset, cp array shape (n_samples, n_features), cada fila es una sample
        K (int): Cantidad de clusters
        max_iter (int, optional): Max Iterations
        tolerance (float, optional): Threshold de convergencia, ya si cambian menos, freno para ahorrar iteraciones

    Returns:
        labels_cp (cp.ndarray): CuPy array of shape (n_samples,), cluster index assigned to each point.
        centroids_cp (cp.ndarray): CuPy array of shape (K, n_features), final centroid coordinates.
    """

    n_samples, n_features = X.shape

    # Centroides Random
    rnd_idxs = cp.random.choice(n_samples, K, replace=False)
    centroids = X[rnd_idxs]

    # tqdm barrirta fachera
    for i in trange(max_iters, desc=f"K-Means (K={K})", unit="iter"):
        '''
        Esto de aca para calcular las distancias usa un truco que se llama 'numpy broadcasting', es mas de la numpy magic que tanto amamos. Medio largo para explicar aca 
        (vean https://numpy.org/doc/stable/user/basics.broadcasting.html y https://numpy.org/doc/stable/user/basics.indexing.html) (y pongan dark mode en la pagina por su bien xd)
        
        pero basicamente, agarra los dos arrays de shapes distintas (fijense que en los Args se ve que X es (n_samples, n_features) y centroids es (K, n_features))
        y usa trucos de numpy para poder restarles y guardar los valores para despues la distancia euclidiana

        bueno, haciendo 'None, ' en el indexing, creas una nueva dimension de tamaño 1, y numpy (again, en este caso cupy)
        automaticamente cuando haces suma o resta, agarra esa dimension y la repite la cantidad de veces del tamaño de la dimension que le toca sumar/restar
        Asi podes restar todos los puntos directamente con cupy, sin tenes que hacer loops, con las famosas operaciones vectorizadas que tienen, y sin error de dimension
        Si no agregás esas dimensiones los shapes serían incompatibles y la resta tiraría error. Y la magia del broadcasting es que no copia datos, solo los repite 
        y el resultado real, queda en la 3ra dimension (index 2 de cada sub-arrray), 
        porque las shapes ahora son (n_samples, 1, n_features) y (1, K, n_features) entonces haces axis=2 para agarrar las distancias
        
        El equivalente a lo que hace la linea es literalmente calcular las distancias, como si hicieras un doble loop 'for', para agarrar cada punto y cada centroide
        y despues restarlos y crear la lista (o array) de las distancias calculandolas una por una, pero eso es muchisimo mas ineficiente, asi que nada, its just numpy magic
        '''

        dists = cp.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2) # again, mucho texto, leer el docstring arriba, numpy magic
        labels = cp.argmin(dists, axis=1) # argmin(dists) para el mas cercano

        # esta es mas straight-forward, solo recalcula los nuevos centroides con el promedio de los puntos que se le asignaron a cada cluster
        # y el if/else es para que si no se le asigno ninguno, lo mantengo igual y listo
        # el vstack agarra los k arrays de tamaño N que genera el for, y los 'stackea' todos en un 2d array de shape (K, N), 
        # con N aca siendo ofc, n_features (tambien conocido como 2, por las 2 columnas del clustering.csv)
        new_centroids = cp.vstack( [X[labels == k].mean(axis=0) if cp.any(labels == k) else centroids[k]   for k in range(K)])

        # mas cupy magic esta vez para ver la convergencia
        # a ver, si explico cada funcion obscura de numpy en detalle no va a entrar la entrega en el max 5mb del campus virtual
        # me quise ir en detalle en la de broadcasting porque como usa el indexing con None me parecio muy interesante
        # pero bueno, esta de allclose basicamente le das una diferencia relativa rtol y una absoluta atol, y compara todos los elementos uno por uno entre los dos arrays de input
        # si son todos iguales (dentro de alguna de las tolerancias) devuelve True, sino False. Asi si encontramos una distribucion de clusters
        # que sea pracrticamente perfecta, no hace falta seguir iterando y ahorro el tiempo
        if cp.allclose(centroids, new_centroids, rtol=rel_tol, atol=abs_tol):
            # print(Fore.GREEN + f"Converged after {i} iterations." + Style.RESET_ALL)
            # print(f"The difference between the centroids is {cp.linalg.norm(centroids - new_centroids):.7f}")
            break

        centroids = new_centroids

    return labels, centroids

# GMM
def gaussian_pdf(x, mean, cov):
    ''' Probability Density Function para Gaussiana '''
    D = x.shape[1]
    cov_det = cp.linalg.det(cov)
    cov_inv = cp.linalg.inv(cov)
    norm_const = 1.0 / cp.sqrt((2 * cp.pi)**D * cov_det)
    x_centered = x - mean
    exponent = -0.5 * cp.sum(x_centered @ cov_inv * x_centered, axis=1)
    return norm_const * cp.exp(exponent)

def initialize_gmm(X, K, centroids):
    ''' Empieza el GMM con los que ya hice del K means '''
    _, features = X.shape
    means = centroids
    covs = cp.array([cp.cov(X.T) + cp.eye(features)*1e-6 for _ in range(K)])
    weights = cp.ones(K) / K
    return means, covs, weights

def expectation(X, means, covs, weights):
    ''' Hace la parte de Expectation de Expectation-Maximization (el γ_ik)'''
    N, K = X.shape[0], means.shape[0]
    gamma = cp.zeros((N, K))

    for k in range(K):
        gamma[:, k] = weights[k] * gaussian_pdf(X, means[k], covs[k])

    gamma_sum = cp.sum(gamma, axis=1, keepdims=True)
    gamma /= gamma_sum
    return gamma

def maximization(X, gamma):
    ''' Maximiza los parametros (μ_k, Σ_k y π_k) '''
    N, D = X.shape
    K = gamma.shape[1]

    N_k = cp.sum(gamma, axis=0)
    weights = N_k / N
    means = cp.dot(gamma.T, X) / N_k[:, cp.newaxis]

    covs = cp.zeros((K, D, D))
    for k in range(K):
        X_centered = X - means[k]
        covs[k] = (gamma[:, k][:, None] * X_centered).T @ X_centered / N_k[k]
        covs[k] += cp.eye(D) * 1e-6  # For numerical stability

    return means, covs, weights

def run_gaussian_mixture(X, k, means, covs, weights, max_iters, rtol, atol):
    ''' Para que el jupyter quede mas limpio, llama directamente a las funcion es de ExpMax'''
    
    for i in trange(max_iters, desc=f"GMM K={k}", unit="iter"):
        gamma = expectation(X, means, covs, weights)
        prev_means, prev_covs, prev_weights = means.copy(), covs.copy(), weights.copy()
        means, covs, weights = maximization(X, gamma)

        if has_converged(means, prev_means, rtol, atol) and \
        has_converged(covs, prev_covs, rtol, atol) and \
        has_converged(weights, prev_weights, rtol, atol):
            break

    return gamma, means, covs, weights

def has_converged(new, old, rtol, atol):
    return cp.allclose(new, old, rtol=rtol, atol=atol)

# DBSCAN
def get_distance_matrix(X):
    # Broadcasting (cupy magic)
    X_norm = cp.sum(X**2, axis=1, keepdims=True)
    D_squared = X_norm + X_norm.T - 2 * X @ X.T
    D_squared = cp.maximum(D_squared, 0) # for numerical stability
    return cp.sqrt(D_squared)


def get_neighbors(dist_matrix, idx, eps):
    ''' Returns los puntos a distancia ε (o menor) del idx'''
    return cp.where(dist_matrix[idx] <= eps)[0]

def expand_cluster(dist_matrix, labels, idx, initial_neighbors, cluster_id, eps, min_pts):
    """
    Expande un cluster desde un nucleo
    """
    labels[idx] = cluster_id
    queue = list(cp.asnumpy(initial_neighbors))  # dynamic list of points to explore
    visited = set(queue)  # fast lookup for deduplication

    while queue:
        n_idx = queue.pop()
        if labels[n_idx] == -1:
            labels[n_idx] = cluster_id
        elif labels[n_idx] == 0:
            labels[n_idx] = cluster_id
            new_neighbors = get_neighbors(dist_matrix, n_idx, eps)
            if len(new_neighbors) >= min_pts:
                for neighbor in cp.asnumpy(new_neighbors):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

def run_dbscan(dist_matrix, eps, min_pts):
    '''
    Hace todo el DBscan

    Args:
        dist_matrix (cp.ndarray): La matriz de distancias de pares (N x N)
        eps (float): Neighborhood radius (ε)
        min_pts (int): La K, puntos minimos por cluster

    Returns:
        cp.ndarray: Las labels (0 = unvisited, -1 = noise, >0 = cluster IDs)
    '''
    N = dist_matrix.shape[0]
    labels = cp.zeros(N, dtype=cp.int32)  # 0 = unvisited
    cluster_id = 0

    iterator = tqdm(range(N), desc=f"DBSCAN (eps={eps}, k={min_pts})", unit="pt")
    for idx in iterator:
        if labels[idx] != 0:
            continue

        neighbor_idxs = get_neighbors(dist_matrix, idx, eps)
        if len(neighbor_idxs) < min_pts:
            labels[idx] = -1
        else:
            cluster_id += 1
            expand_cluster(dist_matrix, labels, idx, neighbor_idxs, cluster_id, eps, min_pts)

    return labels

def run_dbscan_plots(dist_matrix, clust_np, eps_values, min_pts_values, markers):
    fig, axes = plt.subplots(len(eps_values), len(min_pts_values), figsize=(16, 12))

    for i, eps in enumerate(eps_values):
        for j, min_pts in enumerate(min_pts_values):
            labels_cp = run_dbscan(dist_matrix, eps=eps, min_pts=min_pts)
            labels = cp.asnumpy(labels_cp)

            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            n_noise = np.sum(labels == -1)

            ax = axes[i, j]
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

            for idx, (k, color) in enumerate(zip(unique_labels, colors)):
                mask = labels == k
                label = "Noise" if k == -1 else f"Cluster {k}"
                marker = 'x' if k == -1 else markers[idx % len(markers)]
                ax.scatter(clust_np[mask, 0], clust_np[mask, 1],
                            s=8, color=color, label=label, marker=marker)

            ax.set_title(f"eps={eps}, min_pts={min_pts}\nClusters={n_clusters} | Noise={n_noise}")
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.show()


# Reduccion de Dimensiones
def pca(X, n_components):
    ''' PCA con SVD '''
    X_mean = cp.mean(X, axis=0)
    X_centered = X - X_mean

    # SVD
    U, S, Vt = cp.linalg.svd(X_centered, full_matrices=False)
    components = Vt[:n_components]
    X_transformed = X_centered @ components.T

    explained_variance = (S**2) / (X.shape[0] - 1)
    retained_ratio = cp.sum(explained_variance[:n_components]) / cp.sum(explained_variance)

    return X_transformed, components, X_mean, retained_ratio


def pca_reconstruct(X_transformed, components, X_mean):
    return X_transformed @ components + X_mean

def reconstruction_mse(X_original, X_reconstructed):
    ''' Mean Squared Error entre Reconstruccion y Original '''
    return cp.mean((X_original - X_reconstructed) ** 2)



# VAE
import torch
import torch.nn as nn
import torch.nn.functional as F

class VA_Encoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    bce = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld

def train_vae(model, train_loader, val_loader, epochs=30, lr=1e-3, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            loss = vae_loss(recon_x, x, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
