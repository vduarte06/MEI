import matplotlib.pyplot as plt
import numpy as np
import statistics
from random import uniform


# /// Load data 
# m is the number o elements (size of array)
# @returns 2 arrays (m*1) with x data and other with y data
def load_data(file = "ex1data.txt"):
    data = np.loadtxt(file, delimiter=",", unpack=False)
    return data


# Function that calculates h function given a matrix_x (size m*n) and a column vector teta (size n*1)
# n = numero variaveis
# m = numero de dados
# @pre matrix_x(size m*n)
# @pre vector (size n*1)
# @return matrix_h (size m*1)
def h (matrix_x, teta):
    return np.dot(matrix_x, teta)

def h(x, teta):
    x = np.dot(x, teta.T)
    return 1 / (1 + np.exp(-x))

# Returns residual matrix 
def calculate_residual(matrix_x, matrix_teta, matrix_y):
    matrix_h = h(matrix_x, matrix_teta)
    return np.subtract(matrix_h,matrix_y)


# J(tetao, teta1) = (1/2m)* (H - Y)T(H-T)
# H = matrix_x*matrix_teta
def compute_j(m, matrix_x, matrix_teta, matrix_y):
    v = calculate_residual(matrix_x, matrix_teta, matrix_y)
    return (1/(2*m))*(np.dot(v.transpose(), v))

# TETA (s+1) = TETA(S) - a*1/m*(X*TETA - Y)T*X
# @param m size of dataset
# @pre matrix_x(size m*n) - n is the number of variables
# @pre y_vector (size n*1)
# @return matrix_h (size m*1)
def gradient_step(m ,teta_vector, matrix_x, y_vector ,learning_rate):
    matrix_h = h(matrix_x, teta_vector)
    next_teta_vector = teta_vector - learning_rate*(1/m)*(np.dot((matrix_h - y_vector).transpose(), matrix_x))
    return next_teta_vector


def gradient_descent(teta_vector, m, matrix_x, y_vector, learning_rate, alpha = 10**-14 ):
    cost = []
    cost.append(compute_j(m, matrix_x, teta_vector, y_vector))
    next_teta_vector = gradient_step(m, teta_vector, matrix_x, y_vector, learning_rate)
    teta_vector = next_teta_vector
    cost.append(compute_j(m, matrix_x, teta_vector, y_vector))
    error = abs(cost[-1] - cost[-2])  
    while error > alpha:
        next_teta_vector = gradient_step(m, teta_vector, matrix_x, y_vector, learning_rate)
        teta_vector = next_teta_vector
        cost.append(compute_j(m, matrix_x, teta_vector, y_vector))
        error = abs(cost[-1] - cost[-2])      
    return teta_vector, cost



# J(tetao, teta1) = (1/2m)* (H - Y)T(H-Y) + r_factor*(T)^2
# H = matrix_x*matrix_teta 
# T teta vector with n-1 variables (first doesnt count)
# r_factor: regularization factor
# @returns cost of regularization step
def compute_j_regularization(m, matrix_x, matrix_teta, matrix_y, r_factor):
    v = calculate_residual(matrix_x, matrix_teta, matrix_y)
    first_element_reg = np.array([0])
    elements_reg = np.dot(matrix_teta[1:].transpose(), matrix_teta[1:])
    
    regularization_matrix = np.append(first_element_reg, elements_reg)
    regularization_matrix = np.sum(matrix_teta**2)
    j_reg = (1/(2*m))*((np.dot(v.transpose(), v)) + r_factor*regularization_matrix)
    
    return j_reg



# TETA (s+1) = TETA(S)(1-a*r_factor/m) - a*1/m*(X*TETA - Y)T*X
# @step 1 ... n for gradient descent with regularization
# @param m size of dataset
# @pre matrix_x(size m*n) - n is the number of variables
# @pre y_vector (size n*1)
# @r_factor - regularization factor
# @return matrix_h (size m*1)
def gradient_step_regularizatio(m ,teta_vector, matrix_x, y_vector ,learning_rate, r_factor):
    matrix_h = h(matrix_x, teta_vector)
    next_teta0 = teta_vector[0] - learning_rate*(1/m)*(np.dot((matrix_h - y_vector).transpose(), matrix_x[:,0]))
    next_teta_vector = teta_vector[1:]*(1-(learning_rate*r_factor)/m) - learning_rate*(1/m)*(np.dot((matrix_h - y_vector).transpose(), matrix_x[:,1:])) #incluir todas as restantes colunas
    next_teta = np.append(next_teta0, next_teta_vector)
    return next_teta

def gradient_step_regularization(m ,teta_vector, matrix_x, y_vector ,learning_rate, r_factor):
    matrix_h = h(matrix_x, teta_vector)
    next_teta0 = teta_vector[0] - learning_rate*(1/m)*(np.dot((matrix_h - y_vector).transpose(), matrix_x[:,0]))
    next_teta_vector = teta_vector[1:]*(1-(learning_rate*r_factor)/m) - learning_rate*(1/m)*(np.dot((matrix_h - y_vector).transpose(), matrix_x[:,1:])) #incluir todas as restantes colunas
    next_teta = np.append(next_teta0, next_teta_vector)
    return next_teta 

def cost_function(m, theta, x, y, r):
    # Computes the cost function for all the training samples
    total_cost = -(1 / m) * np.sum(
        y * np.log(h(x, theta)) + (1 - y) * np.log(
            1 - h(x, theta)))
    return total_cost


# teta0 (s+1) = teta0(s) - a*1/m*(X*TETA - Y)T*X
# TETA (s+1) = T
def gradient_descent_regularization(teta_vector, m, matrix_x, y_vector, learning_rate, r_factor, alpha = 10**-14):
    cost = []
    cost.append(compute_j_regularization(m, matrix_x, teta_vector, y_vector, r_factor))
    next_teta_vector = gradient_step_regularization(m, teta_vector, matrix_x, y_vector, learning_rate, r_factor)
    teta_vector = next_teta_vector
    #cost.append(compute_j_regularization(m, matrix_x, teta_vector, y_vector, r_factor))
    cost.append(cost_function(m, teta_vector, matrix_x, y_vector, r_factor))
    error = abs(cost[-1] - cost[-2]) 
    print(cost_function(m, teta_vector, matrix_x, y_vector, r_factor))
    # print('cost', error)
    while error > alpha:
        next_teta_vector = gradient_step_regularization(m, teta_vector, matrix_x, y_vector, learning_rate, r_factor)
        teta_vector = next_teta_vector
        cost.append(cost_function(m, teta_vector, matrix_x, y_vector, r_factor))
        error = abs(cost[-1] - cost[-2]) 
    return teta_vector, cost

# /////////////////////////////////////////////////////////////////////////////
# /// Exercicio 2 ///
# /////////////////////////////////////////////////////////////////////////////
# @return root mean square error (same as J) for diferent values of regularization coefficient
def regularization_term_variation(m, matrix_x, matrix_y, teta_vector, learning_rate, reg_term_max):
    teta_gradient_reg = []
    mse = []
    for i in range(reg_term_max):
        teta_gradient_reg, cost_reg_step = gradient_descent_regularization(teta_vector, m , matrix_x, matrix_y, learning_rate, i)
        print(teta_gradient_reg)
        mse.append(cost_reg_step[-1])
    return mse


def main():
    data = load_data("ex1data.txt")
    # /// Shuffle dataset
    #np.random.shuffle(data)
 

# /////////////////////////////////////////////////////////////////////////////
# /// Exercicio 1 ///
# /////////////////////////////////////////////////////////////////////////////
    data_len = len(data)
    training_size = int(0.8*data_len)
    training = data[:training_size]
    test = data[training_size+1 : data_len]

# /////////////////////////////////////////////////////////////////////////////
# /// Exercicio 2 ///
# /////////////////////////////////////////////////////////////////////////////
    x = training[:,0]
    y = training[:,1]

# h0(x)= teta0 + teta1*x
    teta_list = [uniform(y.min(), y.max()), uniform(x.min(), x.max())]
    teta_vector = np.array(teta_list)
    m = len(x)
    matrix_x_train = np.stack((np.ones(m), x), axis=1)
    matrix_y_train = np.array(y)

    learning_rate= 0.01
    teta_gradient, cost = gradient_descent(teta_vector, m, matrix_x_train, matrix_y_train, learning_rate)
    print('Gradiente: Teta0 = {:.4f}, Teta1 = {:.4f}'.format(teta_gradient[0], teta_gradient[1]))
    
    reg_term_max = 1000
    rmse_train = regularization_term_variation(m, matrix_x_train, matrix_y_train, teta_vector, learning_rate, reg_term_max)
    
    # for test data
    m_test = len(test)
    x_test = test[:,0]
    y_test = test[:,1]
    matrix_x_test = np.stack((np.ones(m_test), x_test), axis=1)
    matrix_y_test = np.array(y_test)
    
    rmse_test = regularization_term_variation(m_test, matrix_x_test, matrix_y_test, teta_vector, learning_rate, reg_term_max)
    teta_gradient_reg, cost_reg = gradient_descent_regularization(teta_vector, m , matrix_x_train, matrix_y_train, learning_rate, 100,10**-14)
    # print('Gradiente with regularization : Teta0 = {:.4f}, Teta1 = {:.4f}'.format(teta_gradient_reg[0], teta_gradient_reg[1]))

# PLOT GRAPHS

    plt.figure(0)
    plt.plot(x, y, 'x', color='grey')
    plt.plot(x, teta_gradient[1]*x + teta_gradient[0] , color='black')
    plt.plot(x, teta_gradient_reg[1]*x + teta_gradient_reg[0] , color='red')
    #plt.figure(1)
    #plt.plot(range(reg_term_max), rmse_train, color = 'blue')
    #plt.plot(range(reg_term_max), rmse_test, color = 'red')

    plt.show()


if __name__ == "__main__":
    main()
