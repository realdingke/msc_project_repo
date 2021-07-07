import numpy as np
import matplotlib.pyplot as plt

B = np.array([[4, -2], [-2, 4]])
a = np.array([0, 1]).reshape(2,1)
b = np.array([-2, 1]).reshape(2,1)
c = np.array([-0.6, -0.4]).reshape(2,1)
C = np.array([[3, -2],[-2, 3]])
#define the three functions, input is limited to a 2x1 column vector
f1 = lambda x: x.T@B@x-x.T@x+a.T@x-b.T@x
f2 = lambda x: np.cos((x-b).T@(x-b))+(x-a).T@B@(x-a)
f3 = lambda x: 1 - (np.exp(-(x-a).T@(x-a)) + np.exp(-(x-b).T@B@(x-b)) - 0.1*np.log(np.linalg.det(0.01*np.identity(2)+x@x.T)))

# def grad_f1(x):
#     grad_0 = 6*x[0]-4*x[1]+2
#     grad_1 = 6*x[1]-4*x[0]
#     return np.array([grad_0, grad_1])

def grad_f1(x):
    x_vector = x.reshape(2,1)
    gradient = 2*(x_vector-c).T@C
    return gradient.flatten()

# def grad_f2(x):
#     grad_0 = 4*(2*x[0]-x[1]+1)-2*(x[0]+2)*np.sin((x[0]+2)**2+(x[1]-1)**2)
#     grad_1 = 4*(2*x[1]-x[0]-2)-2*(x[1]-1)*np.sin((x[0]+2)**2+(x[1]-1)**2)
#     return np.array([grad_0, grad_1])

def grad_f2(x):
    x_vector = x.reshape(2,1)
    gradient = -2*np.sin((x_vector-b).T@(x_vector-b))@(x_vector-b).T + 2*(x_vector-a).T@B
    return gradient.flatten()

# def grad_f3(x):
#     x_vector = x.reshape(2,1)
#     gradient = 2*np.exp(-(x_vector-a).T@(x_vector-a))@(x_vector-a).T 
#     + 2*np.exp(-(x_vector-b).T@B@(x_vector-b))@(x_vector-b).T@B 
#     + (20/(100*(x[0]**2+x[1]**2)+1))*x_vector.T
#     return gradient.flatten()

def grad_f3(x):
    x_vector = x.reshape(2,1)
    gradient = 2*np.exp(-(x_vector-a).T@(x_vector-a))@(x_vector-a).T + 2*np.exp(-(x_vector-b).T@B@(x_vector-b))@(x_vector-b).T@B + (20/(100*(x[0]**2+x[1]**2)+1))*x_vector.T
    return gradient.flatten()

def gradient_descent(function, grad_function, starting_point, step_size, iterations):
    #initialize the input matrix with desired dimensions
    x_inputs = np.zeros(iterations*2).reshape(iterations,2,1)
    x_inputs[0] = starting_point.reshape(2,1)
    for i in range(iterations):
        if i > 0:
            x_inputs[i] = x_inputs[i-1] - step_size*grad_function(x_inputs[i-1].reshape(2,)).reshape(2,1)
    final_input = x_inputs[-1]
    local_minimum = function(x_inputs[-1]).sum()
    print(f"local minimum of the function is found to be {np.round_(local_minimum,4)}, occuring at the input x of {np.round_(final_input,4)}.")
#     trace = []
#     for array in x_inputs:
#         trace.append((array.reshape(2,)[0], array.reshape(2,)[1]))
    return x_inputs

def plot_contours(function, inputs_array, multiplier=1.5, function_name='the function'):
    x1 = np.linspace(-inputs_array[-1][0] * multiplier, inputs_array[-1][0] * multiplier, 100)
    x2 = np.linspace(-inputs_array[-1][1] * multiplier, inputs_array[-1][1] * multiplier, 100)
    X1, X2 = np.meshgrid(x1, x2)
    Y = np.zeros(shape=(x1.size, x2.size))
    for i, val1 in enumerate(x1):
        for j, val2 in enumerate(x2):
            Y[i,j] = function(np.array([val1, val2]).reshape(2,1)).sum()
    #get the trace of all steps of GD
    trace = []
    for array in inputs_array:
        trace.append((array.reshape(2,)[0], array.reshape(2,)[1]))
    fig0 = plt.figure()
    plt.contourf(X1, X2, Y, alpha=0.7)
    CS = plt.contour(X1, X2, Y, linestyles='dashed', linewidths=1, colors='black')
    plt.clabel(CS, inline=1, fontsize=8)
    fig0.gca().add_patch(plt.Polygon(trace, closed=None, fill=None, edgecolor='y'))
    plt.scatter(inputs_array[-1][0], inputs_array[-1][1], c='red',s=30)   #annotate the end point with a red dot
    plt.title(f"Contour plot of gradient descent for {function_name}, found minimum is {np.round_(function(inputs_array[-1]).sum(), 5)}")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()

#plot_contours(f1, gradient_descent(f1, grad_f1, np.array([0.3,0]), 0.15, 50))
#plot_contours(f2, gradient_descent(f2, grad_f2, np.array([0.3,0]), 0.07, 50), multiplier=2, function_name='f2')
#plot_contours(f3, gradient_descent(f3, grad_f3, np.array([0.3,0]), 0.885, 50), multiplier=12, function_name='f3')


##analysis for f2
#stepsizes = [0.01,0.06,0.095,0.145]
#inputs_f2 = []
#for stepsize in stepsizes:
#    inputs_f2.append(gradient_descent(f2, grad_f2, np.array([0.3,0]), stepsize, 50))
#    
#fig = plt.figure()
#for num, inputs_array in enumerate(inputs_f2):
#    x1 = np.linspace(-inputs_array[-1][0] * 2, inputs_array[-1][0] * 2, 100)
#    x2 = np.linspace(-inputs_array[-1][1] * 2, inputs_array[-1][1] * 2, 100)
#    X1, X2 = np.meshgrid(x1, x2)
#    Y = np.zeros(shape=(x1.size, x2.size))
#    for i, val1 in enumerate(x1):
#        for j, val2 in enumerate(x2):
#            Y[i,j] = f2(np.array([val1, val2]).reshape(2,1)).sum()
#    #get the trace of all steps of GD
#    trace = []
#    for array in inputs_array:
#        trace.append((array.reshape(2,)[0], array.reshape(2,)[1]))
#    ax = fig.add_subplot(2, 2, num+1)
#    ax.contourf(X1, X2, Y, alpha=0.7)
#    CS = ax.contour(X1, X2, Y, linestyles='dashed', linewidths=1, colors='black')
#    ax.clabel(CS, inline=1, fontsize=8)
#    ax.scatter(inputs_array[-1][0], inputs_array[-1][1], c='red',s=20)   #annotate the end point with a red dot
#    ax.set_title(f'Using step size {stepsizes[num]}, found minimum is {np.round_(f2(inputs_array[-1]).sum(), 5)}')
#    fig.gca().add_patch(plt.Polygon(trace, closed=None, fill=None, edgecolor='y'))
#
#fig.show()

##analysis for f3
#stepsizes = [0.01,0.15,0.89,0.95]
#inputs_f3 = []
#for stepsize in stepsizes:
#    inputs_f3.append(gradient_descent(f3, grad_f3, np.array([0.3,0]), stepsize, 50))
#    
#fig = plt.figure()
#for num, inputs_array in enumerate(inputs_f3):
#    x1 = np.linspace(-inputs_array[-1][0] * 8, inputs_array[-1][0] * 8, 100)
#    x2 = np.linspace(-inputs_array[-1][1] * 8, inputs_array[-1][1] * 8, 100)
#    X1, X2 = np.meshgrid(x1, x2)
#    Y = np.zeros(shape=(x1.size, x2.size))
#    for i, val1 in enumerate(x1):
#        for j, val2 in enumerate(x2):
#            Y[i,j] = f3(np.array([val1, val2]).reshape(2,1)).sum()
#    #get the trace of all steps of GD
#    trace = []
#    for array in inputs_array:
#        trace.append((array.reshape(2,)[0], array.reshape(2,)[1]))
#    ax = fig.add_subplot(2, 2, num+1)
#    ax.contourf(X1, X2, Y, alpha=0.7)
#    CS = ax.contour(X1, X2, Y, linestyles='dashed', linewidths=1, colors='black')
#    ax.clabel(CS, inline=1, fontsize=8)
#    ax.scatter(inputs_array[-1][0], inputs_array[-1][1], c='red',s=20)   #annotate the end point with a red dot
#    ax.set_title(f'Using step size {stepsizes[num]}, found minimum is {np.round_(f3(inputs_array[-1]).sum(), 5)}')
#    fig.gca().add_patch(plt.Polygon(trace, closed=None, fill=None, edgecolor='y'))
#
#fig.show()

if __name__=="__main__":
    print('Result of gradient descent for f2 with step size 0.07:')
    _ = gradient_descent(f2, grad_f2, np.array([0.3,0]), 0.07, 50)
    print()
    print('Result of gradient descent for f3 with step size 0.885:')
    _ = gradient_descent(f3, grad_f3, np.array([0.3,0]), 0.885, 50)
