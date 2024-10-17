import autograd.numpy as np
from autograd import grad

# Definir la función objetivo
def f(xy):
    x, y = xy
    r = (x**2 + y**2)
    return r**0.25 * (np.sin(50 * r**0.1)**2 + 1)

# Gradiente automático con Autograd
gradient_f = grad(f)

# Gradiente descendente usando el gradiente de Autograd
def gradient_descent_2d_autograd(x0, y0, learning_rate, tolerance, max_iter=10000):
    xy = np.array([x0, y0])
    for i in range(max_iter):
        grad_xy = gradient_f(xy)
        if np.linalg.norm(grad_xy) < tolerance:
            break
        xy -= learning_rate * grad_xy
        if((xy[0]>100 or xy[0]<-100)or(xy[1]>100 or xy[1]<-100)):
            xy += learning_rate * grad_xy
    return xy[0], xy[1]

# Parámetros iniciales
x0, y0 = np.random.uniform(-100, 100, 2)  # Puntos iniciales aleatorios
learning_rate = 0.01                      # Tasa de aprendizaje
tolerance = 1e-6                         # Tolerancia para detener

# Ejecutar gradiente descendente
min_x, min_y = gradient_descent_2d_autograd(x0, y0, learning_rate, tolerance)

print(f"El valor mínimo de x es: {min_x}")
print(f"El valor mínimo de y es: {min_y}")
print(f"El valor mínimo de la función es: {f([min_x, min_y])}")
