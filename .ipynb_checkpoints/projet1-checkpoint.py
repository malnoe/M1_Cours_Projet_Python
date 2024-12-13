import numpy as np
from functions import Function1, Function2
from baseloss import LossFunction
from baseoptimizer import Optimizer
from utils import make_random_func1, make_random_func2
from time import process_time

def gradient_descent(
    eps: float, theta0: np.array, L: Function1, eta: float
) -> (np.ndarray, float):
    """Perform gradient descent optimization."""
    # Initialisation des variables.
    theta = theta0
    liste_theta = [theta0]

    while True:
        # Calcul du gradient et du nouveau theta associe.
        grad = Function1.grad_oracle(L, theta)
        theta_new = theta - eta * grad
        liste_theta.append(theta_new)

        # Verification de la condition de convergence.
        if np.linalg.norm(theta_new - theta) <= eps:
            break

        # Ecrasement de l'ancienne valeur de theta.
        theta = theta_new

    print(f"Nombre d'étapes : {len(liste_theta)}")
    return theta, len(liste_theta)

def gradient_descent_1(eps, theta0, L, eta):
    """Perform gradient descent optimization, returns the number of steps."""
    # Initialisation des variables.
    theta = theta0
    liste_theta = [theta0]

    while True:
        # Calcul du gradient et du nouveau theta associe.
        grad = Function1.grad_oracle(L, theta)
        theta_new = theta - eta * grad
        liste_theta.append(theta_new)

        # Verification de la condition de convergence.
        if np.linalg.norm(theta_new - theta) <= eps:
            break

        # Ecrasement de l'ancienne valeur de theta.
        theta = theta_new

    return len(liste_theta)

def newton_descent_naive(
    eps: float, theta0: np.array, L: Function1, eta: float
) -> (np.ndarray, float):
    """Perform naive Newton descent optimization."""

    # Initialisation des variables.
    theta = theta0
    liste_theta = [theta0]

    while True:
        # Calcul de la hessienne et du gradient.
        hessian_inv = np.linalg.inv(Function1.hessian_oracle(L, theta))
        grad = Function1.grad_oracle(L, theta)
        theta_new = theta - eta * np.dot(hessian_inv, grad)
        liste_theta.append(theta_new)

        # Verification de la condition de convergence.
        if np.linalg.norm(theta_new - theta) <= eps:
            break

        # Ecrasement de l'ancienne valeur de theta.
        theta = theta_new

    print(f"Nombre d'étapes : {len(liste_theta)}")
    return theta, len(liste_theta)

def newton_descent_naive_1(
    eps: float, theta0: np.array, L: Function1, eta: float
) -> (np.ndarray, float):
    """Perform naive Newton descent optimization."""

    # Initialisation des variables.
    theta = theta0
    liste_theta = [theta0]

    while True:
        # Calcul de la hessienne et du gradient.
        hessian_inv = np.linalg.inv(Function1.hessian_oracle(L, theta))
        grad = Function1.grad_oracle(L, theta)
        theta_new = theta - eta * np.dot(hessian_inv, grad)
        liste_theta.append(theta_new)

        # Verification de la condition de convergence.
        if np.linalg.norm(theta_new - theta) <= eps:
            break

        # Ecrasement de l'ancienne valeur de theta.
        theta = theta_new

    return len(liste_theta)

def newton_descent_clever(theta0: np.array, L: Function1) -> np.ndarray:
    """Perform direct computation for Newton descent."""
    theta = theta0 - np.dot(np.linalg.inv(L.A), L.b)
    return theta

def bfgs_descent(eps: float, theta0: np.array, L: Function1, eta: float):
    """Perform BFGS optimization."""
    # Initialisation des variables
    n = len(theta0)
    B = np.eye(n)
    grad = Function1.grad_oracle(L, theta0)
    p = -eta * grad
    theta1 = theta0 + p
    liste_theta = [theta0, theta1]

    while np.linalg.norm(liste_theta[-1] - liste_theta[-2]) > eps:
        grad_new = Function1.grad_oracle(L, liste_theta[-1])
        theta_new = liste_theta[-1] - eta * np.dot(B, grad)
        s = theta_new - liste_theta[-1]
        y = grad_new - grad
        Bs = np.dot(B, s)
        B += (np.outer(y, y) / np.dot(y, s)) - (np.outer(Bs, Bs) / np.dot(s, Bs))

        liste_theta.append(theta_new)
        grad = grad_new

    print(f"Nombre d'étapes : {len(liste_theta)}")
    return liste_theta[-1]

def bfgs_descent_1(eps: float, theta0: np.array, L: Function1, eta: float):
    """Perform BFGS optimization."""
    # Initialisation des variables
    n = len(theta0)
    B = np.eye(n)
    grad = Function1.grad_oracle(L, theta0)
    p = -eta * grad
    theta1 = theta0 + p
    liste_theta = [theta0, theta1]

    while np.linalg.norm(liste_theta[-1] - liste_theta[-2]) > eps:
        grad_new = Function1.grad_oracle(L, liste_theta[-1])
        theta_new = liste_theta[-1] - eta * np.dot(B, grad)
        s = theta_new - liste_theta[-1]
        y = grad_new - grad
        Bs = np.dot(B, s)
        B += (np.outer(y, y) / np.dot(y, s)) - (np.outer(Bs, Bs) / np.dot(s, Bs))

        liste_theta.append(theta_new)
        grad = grad_new

    return len(liste_theta)


def stochastic_descent(
    eps: float, theta0: np.ndarray, L: Function2, eta: float, batch_size: int
) -> (np.ndarray, float):
    """Perform stochastic gradient descent."""
    # Initialisation des variables.
    theta = theta0
    liste_theta = [theta0]
    n_points = L.X.shape[0]

    while True:
        # Calcul des gradients aleatoire et de leur moyenne.
        batch_indices = np.random.choice(n_points, size=batch_size, replace=False)
        grad = L.batched_grad_oracle(batch_indices, theta)
        grad_mean = np.mean(grad, axis=0)

        # Calcul du nouveau theta associe.
        theta_new = theta - eta * grad_mean
        liste_theta.append(theta_new)

        # Verification de la condition de convergence
        if np.linalg.norm(theta_new - theta) <= eps:
            break

        # Ecrasement de l'ancienne valeur de theta.
        theta = theta_new

    return theta, len(liste_theta)

def adam(
    L, theta_0, eps, eta, beta_1, beta_2, epsilon=1e-8
) -> np.ndarray:
    """Perform Adam optimization."""
    # Initialisation des variabels
    m = np.zeros_like(theta_0)
    v = np.zeros_like(theta_0)
    theta = theta_0
    liste_theta = [theta_0]
    n_points = L.X.shape[0]
    batch_size = len(theta_0)

    while True:
        # Calcul des des gradients aleatoires et de leur moyenne
        batch_indices = np.random.choice(n_points, size=batch_size, replace=False)
        grad = L.batched_grad_oracle(batch_indices, theta)
        grad_mean = np.mean(grad, axis=0)

        # Calcul des variables (moments) permettant de controler la descente.
        m = beta_1 * m + (1 - beta_1) * grad_mean
        v = beta_2 * v + (1 - beta_2) * grad_mean**2
        m_hat = m / (1 - beta_1)
        v_hat = v / (1 - beta_2)
        v_hat = np.maximum(v_hat, epsilon)

        # Calcul du nouveau theta associe.
        theta_new = theta - eta * m_hat / (np.sqrt(v_hat) + epsilon)
        liste_theta.append(theta_new)
        
        # Verification de la condition de convergence.
        if np.linalg.norm(theta_new - theta) <= eps:
            break

        # Ecrasement de l'ancienne valeur de theta.
        theta = theta_new
        

    return theta, len(liste_theta)

def processing_time(function):
    """calculate function processing time"""
    # commence le compteur
    start = process_time()
    # execute la fonction
    function
    # stop le compteur
    end = process_time()
    #calcul de la duree en millieme de seconde
    elapsed = (end - start)*1000
    return elapsed
