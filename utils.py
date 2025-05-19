import cupy as cp
import numpy as np
import time
from tqdm import trange
from colorama import Fore, Style

# para plotear las gaussianas
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

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

# def compute_gmm_L(X, cluster_assignments, means):
#     ''' mismo concepto que antes, no hace falta repetir'''
#     L = 0.0
#     for k in range(means.shape[0]):
#         cluster_points = X[cluster_assignments == k]
#         if cluster_points.shape[0] > 0:
#             dists = cp.linalg.norm(cluster_points - means[k], axis=1)
#             L += cp.sum(dists)
#     return float(L)


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


