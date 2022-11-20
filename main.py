import random
import time
from sys import exit
import numpy as np
import sympy as sp
from sympy import *
import os
import scipy.linalg as spl
import matplotlib.pyplot as plt
from numpy import linalg as LA, matrix
import scipy.optimize as op

# jarrabt el hessienne wel grad yekhdmou maa l graph wel l de niveau jawna ahla jaw
x, y = symbols('x y', real=True)
func = None
f1 = None
string_func = ""
A = None
B = None
eps = None
Matrix = false
rosen = false


def functions():  # HAWEL TCHOUF CHNIA MOCHKOLT L MENU HEDHA
    """ qu'elle fonction voulez vous choisir? """
    while True:
        os.system('clear')
        # output.clear()
        print("Choisissez l'option que vous voulez utilisé [1-3]: ")
        print("""
           1 : f1(x,y) = (1-x) ** 2 + 100 * ( y - (x ** 2) ** 2)"
           2 : f2 = None  #
           3 : f3(x,y) = x * exp (-x**2 - y**2)"""
              )
        choix = input("\nEntrez votre choix [1-3] : ")
        global func, x
        if choix == '1':
            func = (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
            global rosen
            rosen = true
            break
        elif choix == '2':
            global A, B
            A = np.array([[3, -1, 0, 0, 0],
                          [-1, 12, -1, 0, 0],
                          [0, -1, 24, -1, 0],
                          [0, 0, -1, 48, -1],
                          [0, 0, 0, -1, 96]])
            B = np.array([1, 2, 3, 4, 5])
            # x = matrix[x]
            # func = 0.5 * x @ A @ x - x @ B
            # TODO how to draw this ?? abir ??
            break

        elif choix == '3':
            func = x * exp(-x ** 2 - y ** 2)  # TODO to change with quadratic polynomial : Manel
            break
        else:
            print("choix incorrecte")
            functions()
            break


#         # os.clear()
# output.clear()
# exit()


def entree():
    # instance les variables pour le fonctionnement de eval
    # nbvariable = int(input("Donner le nombre de variables"))
    global string_func
    # string_func = input("""Votre fonction est sous la forme f(x,y)=ax**2+by**2+cxy:
    #                         sachant que x**2 est x au carrée \nentez les coeff a,b,c séparées par virgule',' """).lower()
    a, b, c = [int(x) for x in input("""Votre fonction est sous la forme f(x,y)=ax**2+by**2+cxy:
                            sachant que x**2 est x au carrée \nentez les coeff a,b,c séparées par virgule',' """).split(
        ', ')]
    # X = extract_symbols(string_func, nbvariable)
    # fct = lambda x, y: string_func
    global func, x, y, rosen

    # func = eval(string_func)
    func = a * x ** 2 + b * y ** 2 + c * x * y
    rosen = false
    print(func)
    # TODO  force user to enter quadratic form


# a=input()
# b=input()
# c=input()
# func = a*x**2+ b*y**2 + c*x*y


def entrer_matrice():
    n = int(input("Entrer le dimension de la matrice"))
    global A
    global B
    global eps
    A = np.zeros((n, n))
    B = np.zeros((n, 1))
    for r in range(0, n):
        for c in range(0, n):
            A[(r), (c)] = (input("Element a[" + str(r + 1) + "," + str(c + 1) + "] "))

    for i in range(0, n):
        B[(i)] = (input('b[' + str(i + 1) + ']: '))

    # for j in range(0, 2):
    #     B[(j)] = (input('X0[' + str(j + 1) + ']: '))
    #
    # eps = int(input("entrer la tolérance "))

    # global func
    # func = eval(A)


# TODO conversion de A et B : Manel
# A=hessienne
# B=b()

def choix_entree():
    """ Affiche le meneu d choix de saisie """
    while True:
        # output.clear()

        print("Choisissez l'option que vous voulez utilisé [1-2]: ")
        print("""
               1 : saisir une fonction polynomiale.
               2 : Saisir une fonction matricielle.
               3 : Saisir une fonction de rosenbork.
               3 : revenir au menu précédant."""
              )

        # TODO add rosenbrok type:Manel
        choix = input("\nEntrez votre choix [1-3] : ")
        if choix == '1':
            entree()
            niveau_2()
        if choix == '2':
            entrer_matrice()
            niveau_2()
        if choix == '3':
            global rosen
            rosen = true
            # TODO manel add rosen here
        elif choix == '4':
            main()
            break
        else:
            print("choix incorrecte")
            choix_entree()
        #         # os.clear()

        # output.clear()
        exit()


def main():
    """ Affiche le menu principale du programme qui contient le choix de l'utilisateur"""
    while True:
        # output.clear()

        print("Choisissez l'option que vous voulez utilisé [1-3]: ")
        print("""
            1 : Choisir une fonction de la mémoire
            2 : Saisir sa propre fonction.
            3 : quitter"""
              )
        choix = input("\nEntrez votre choix [1-3] : ")
        if choix == '1':
            functions()
            niveau_2()
        if choix == '2':
            choix_entree()
            niveau_2()
        elif choix == '3':
            break
        else:
            print("choix incorrecte")
            main()
        # os.clear()

        # output.clear()
        exit()


def graph(func):
    x, y = sp.symbols('x y', real=True)
    v1 = var('x y')
    a = np.linspace(-400, 400, 100)
    b = np.linspace(-400, 400, 100)
    X, Y = np.meshgrid(a, b)
    print(func)
    f = lambdify([x, y], func, "numpy")
    Z = f(X, Y)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, rstride=5, cstride=5,
                    cmap='cool')
    ax.set_title("Surface Bonus", fontsize=13)
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('y', fontsize=11)
    ax.set_zlabel('Z', fontsize=11)
    plt.show()
    return ax


def is_pos_def(x):
    """symetrique positive ou nn"""
    return (np.array_equal(A, np.transpose(A))) and (np.all(np.linalg.eigvals(A) > 0)) and (spl.det(A) != 0)


def graph_niv(func, ax):
    x, y = sp.symbols('x y', real=True)
    v1 = var('x y')
    a = np.linspace(-400, 400, 100)
    b = np.linspace(-400, 400, 100)
    X, Y = np.meshgrid(a, b)
    print(func)
    f = lambdify([x, y], func, "numpy")
    Z = f(X, Y)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, rstride=5, cstride=5,
                    cmap='cool')
    ax.set_title("Surface Bonus", fontsize=13)
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('y', fontsize=11)
    ax.set_zlabel('Z', fontsize=11)
    ax.contour(X, Y, Z, 10, cmap="autumn_r", linestyles="solid", offset=-1)
    ax.contour(X, Y, Z, 10, colors="k", linestyles="solid")
    ax.view_init(20, 70)
    plt.show()
    # x, y = sp.symbols('x y', real=True)
    # v1 = var('x y')
    # X, Y = np.meshgrid(np.linspace(0, 2, 201), np.linspace(0, 2, 201))
    # f = lambdify([x, y], func, "numpy")
    # Z = f(X, Y)
    # ax.contour(X, Y, Z, np.linspace(0, 1, 21))
    # plt.colorbar()
    # plt.show()


class bcolors:  # Affichage avec couleurs.
    OK = '\033[92m'  # vert
    RG = '\033[91m'  # rouge
    RESET = '\033[0m'  # rest des couleurs


def conjugue(A, b, X, itMax, tol, pas=0):
    if (is_pos_def(A) == False):
        raise ValueError("Matrice A n'est pas symetrique positive")
    else:
        R = b - A.dot(X)  # -gradient de f(Xk)
        P = R  # Direction initiale (-gradient de la fonction)
        k = 0
        alpha = pas
        start = time.time()
        while (k <= itMax) and (LA.norm(R) > tol):  # verification des condition:
            # && nombre d'itération ne dépasse pas nbr max
            Ap = A.dot(P)  # A * P
            if pas == 0:  # cas optimal
                alpha = np.transpose(R).dot(R) / np.transpose(P).dot(Ap)  # pas optimale
            X = X + (alpha * P)  # X(k+1) = X(k) + direction(k) * pas(k) TODO also this :eya
            Rancien = R  # R(k) -->gradient f(k+1)
            R = R - (alpha * Ap)  # R(k+1) --> -gradient f(k+1)

            beta = (np.transpose(R).dot(R) / np.transpose(Rancien).dot(Rancien))
            P = R + beta * P  # direction k+1 TODO check how to draw this in the graph: eya

            k = k + 1  # incrémentation d'itération
        # print("\nle nombre d'itération = \n", k)
        #
        # print(bcolors.OK + "\n notre solution minimale cherchée X = \n", X)
    end = time.time()
    duree = end - start
    # return result , nb it , exec duration
    return X, k, duree


def graph_niveau(func):
    x, y = sp.symbols('x y', real=True)
    v1 = var('x y')
    xlist = np.linspace(-3.0, 3.0, 100)
    ylist = np.linspace(-3.0, 3.0, 100)
    X, Y = np.meshgrid(xlist, ylist)
    f = lambdify([x, y], func, "numpy")
    Z = f(X, Y)
    fig, ax = plt.subplots(1, 1)
    cp = ax.contour(X, Y, Z)
    fig.colorbar(cp)  # Add a colorbar to a plot
    ax.set_title('Graph de niveau')
    # ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')
    plt.show()


def grad(f):
    # x,y=symbols('x y',real=True)
    # f = func
    tab = [diff(f, x), diff(f, y)]
    return tab


def hessienne(f):
    params = [x, y]
    gradd = grad(f)
    tab = np.array([[None] * 2, [None] * 2])
    for i in range(2):
        for j in range(2):
            tab[i, j] = diff(gradd[i], params[j])
    return tab


def comparatif(A, B):
    # list = [[0, 0, 0], [1, 2, 3], [3, 3, 4]]
    # results = np.zeros((len(A), 1))
    # b = np.array([[3.], [2.], [3.]])  # b=vecteur colonne (0 1)
    # X0 = np.array([[0.], [0.], [0.]])  # x0=vecteur colonne (0 0)
    # a = np.array([[2., 0., 1], [0., 2., 0.], [1., 0., 2.]])  # A=matrice carré
    tol = 1e-5  # La précision fixée à 10e-5
    if rosen:
        x, y = sp.symbols('x y', real=True)
        f = lambdify([(x, y)], func, "numpy")
        list = [[0, 0], [1, 2], [2, 3]]  # TODO to check with omar
        for i in [0, 1, 2]:
            # z = f(x, y)
            x0 = random.choice(list)
            X = x0[0]
            Y = x0[1]

            # grad = rosenbrock_grad([X, Y])
            # chercher la direction initial ali hiya -grad(f(x0))
            # d0 = [-grad[0], -grad[1]]
            print("comparatif numero ", i, "avec x0 = ", x0)
            # print("direction initial d0 = ", d0)
            print(op.fmin_cg(f, (x0[0], x0[1])))
            print("\n")
    else:
        for i in range(0, 3):
            x0 = np.random.random_sample(size=(len(A), 1))
            x0.astype(int)
            A.astype(int)
            B.astype(int)
            B = B.T
            X, k, du = conjugue(A, B, x0, 100, 1.5e-8)
            print("la resultat n°\n", i, "=", X, "avec un vecteur de depart x0 :", x0, "et une durée d'exec = ", du,
                  "et un nbr exec ", k)


# TODO add niveau 3
def niveau_2():
    """ qu'elle traitement voulez vous effectuer?"""
    while True:
        # os.system('clear')
        # output.clear()
        print("Choisissez l'option que vous voulez utilisé [1-7]: ")
        print("""
            1 : Tracer la courbe
            2 : Tracer les lignes de niveaux et les ajouter au graphe existant 
            3 : Calculer le vecteur gradient
            4 : Calculer la matrice Hessienne
            5 : Appliquer la méthode de gradient conjugué standard
            6 : Visualiser un comparatif avec 3 différents X0 (le vecteur de départ)
            7 : Revenir au niveau 1"""
              )
        choix = input("\nEntrez votre choix [1-7] : \n")
        if choix == '1':
            graph(func)
        elif choix == '2':
            ax = graph(func)
            graph_niv(func, ax)
        elif choix == '3':
            print("le gradiant est ")
            print(grad(func))
        elif choix == '4':
            print("l'hessienne est ")
            print(hessienne(func))
        # elif choix == '5':
        # TODO redirection vers niv 3
        # conjugue(A, b, X, itMax, tol)
        elif choix == '6':
            comparatif(A, B)
            # TODO test (Rosenbrok)
        elif choix == '7':
            entree()
            break
        else:
            print("choix incorrecte")
            niveau_2()
        # output.clear()
        #         # os.clear()
        exit()


# TODO

if __name__ == '__main__':
    main()

    # x0 = np.random.random_sample(size=(1,len(A)))
    #         x, k, du = conjugate_gradient_mat(A, B, 1.5e-8, x0)
    #         print("la resultat n°", i, "=", x, "avec un vecteur de depart :", k, "et une durée d'exec = ", du)

# def menu_fonctions()
# def extract_symbols(string_func, nbS):
#     i = 0
#     s = ''
#     while i < nbS:
#         s += 'x' + str(i) + ' '
#         i += 1
#     return symbols(s, real=True)


# vecteur gradient
# def gradient(string_func, nbS=2):
#     X = [x, y]
#     func = eval(string_func)
#     i = 0
#     dX = [None] * nbS
#     while i < nbS:
#         dX[i] = diff(func, X[i])
#         i += 1
#     return dX


# matrice hessienne
# def hessienne(dX, string_func):
#     nbS = len(dX)
#     i = 0
#     X = extract_symbols(string_func, nbS)
#     H = np.array([[None] * nbS, [None] * nbS])
#     while i < nbS:
#         j = 0
#         while j < nbS:
#             H[i, j] = diff(dX[i], X[j])
#             j += 1
#         i += 1
#     return H

# def graph():
#    fig = plt.figure(figsize=(8, 8))
#    ax = plt.axes(projection='3d')
#    # ax.grid()
#    global x
#    global y
#    print(func)
#    t = func
#    X = np.linspace(-1, 1, 100)
#    Y = np.linspace(-1, 1, 100)
#    x, y = np.meshgrid(X, Y)
# t = f(x, y)
#    ax.plot_surface(x, y, t, rstride=1, cstride=1, cmap='viridis', edgecolor='none')#TODO houni l'erreur apparently MANEL/EYA
# ax.contour3D(x, y, t, 50, cmap='binary')
# ax.plot3D(x, y, t)
# ax.plot_surface(x, y,t, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
#    ax.set_title('3D Parametric Plot')

# Set axes label
#    ax.set_xlabel('x', labelpad=20)
#    ax.set_ylabel('y', labelpad=20)
#    ax.set_zlabel('t', labelpad=20)

#    plt.show()


# def graph_niveau():
#     # Tracer le graphe
#     ax = np.linspace(-1, 1, 100)
#     ay = np.linspace(-1, 1, 100)
#     ax, ay = np.meshgrid(ax, ay)
#     # fn =  simpledialog.askstring("Input","fonction")
#     print(eval(string_func))
#     Z = eval(string_func)
#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')
#     print(cos(10) + sin(10))
#     # ax.plot_surface(x, y, Z)
#     ax.plot_surface(ax, ay, Z, cmap=cm.nipy_spectral_r)
#     msg = "Graphe du fonction  " + func
#     plt.title(msg)
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('Z')
#     # plt.legend("Graphe du fontion")
#     plt.figure(2)
#     plt.axis('equal')
#     # plt.contourf(x, y, Z, 20)
#     plt.contour(ax, ay, Z, cmap=cm.nipy_spectral_r)
#     plt.colorbar()
#     plt.show()
# """"
# A = np.array([[3, -1,0,0,0], [-1, 12,-1,0,0],[0,-1, 24,-1,0],[0,0,-1,48,-1],[0,0,0,-1,96]])
# b = np.array([1,2,3,4,5])
# x0 = np.array([0, 5])
# """
# A = np.array([[2, 0,1], [0, 2,0],[1,0, 2]])
# b = np.array([3,2,3])
# x0 = np.array([0, 0])
#
#
# eps = 1e-2  # condition d'arret ba3d nda5louha men 3and il utilisateur
#
#
#
# x=conjugate_gradient_matrice(A, b, eps)
# print(x)


# def entrer_matrice(A,B,n):
#     for r in range(0,n):
#         for c in range(0,n):
#             A[(r),(c)]=(input("Element a["+str(r+1)+","+str(c+1)+"] "))
#         B[(r)]=(input('b['+str(r+1)+']: '))


# def conjugate_gradient_matrice(A, b, eps, alpha):
#     """ calcule le gradiant conjugée de A et B
#         cette fonction retourne la resultat x, le nbre d'iterations k et le temps d'execution  duree
#         exemple   x,k,duree=conjugate_gradient_matrice(A,B)
#     """
#     if (is_pos_def(A) == False) | (A != A.T).any():
#         raise ValueError('Matrice A n\est pas symetrique positive   ')
#     d0 = b
#     k = 0
#     x = np.zeros(A.shape[-1])  # initialisation mta3 il solution
#     nb_max_iter = 100  # Nb max d'iteration
#     start = time.time()
#     """" Condition d'arret"""
#     while LA.norm(d0) > eps and k < nb_max_iter:
#         print("Iter", k)
#
#         if k == 0:
#             direc = d0  # direction initiale
#             # print(direc)
#         else:
#             Belta = - (direc @ A @ d0) / (direc @ A @ direc)
#             # print("Belta: " + str(Belta))
#             direc = d0 + Belta * direc
#             # print("direction: " + str(direc))
#             # print("norm: " + str(np.linalg.norm(direc)))
#         # alpha = (direc @ d0) / (direc @ A @ direc)  # à vérifier si elle est le pas ou nn si nn on la met une cte 1/2
#
#         print("alpha=", alpha)
#         x = x + alpha * direc
#         print("newx: " + str(x))
#
#         d0 = d0 - alpha * (A @ direc)
#         k = + 1
#     end = time.time()
#     duree = end - start
#     return x, k, duree


# def conjugate_gradient_mat(A, b, eps, x):
#     """ calcule le gradiant conjugée de A et B
#             cette fonction retourne la resultat x, le nbre d'iterations k et le temps d'execution  duree
#             exemple   x,k,duree=conjugate_gradient_matrice(A,B)
#         """
#     if (is_pos_def(A) == False) | (A != A.T).any():
#         raise ValueError("Matrice A n'est pas symetrique positive")
#     d0 = b - A @ x
#     k = 0
#     nb_max_iter = 100  # Nb max d'iteration
#     start = time.time()
#     while LA.norm(d0) > eps and k < nb_max_iter:
#         direc = d0
#         q = A @ direc
#         alpha = (direc @ d0) / (direc @ q)
#         # print("alpha=", alpha)
#         x = x + alpha * direc
#         # print("newx: " + str(x))
#         d0 = d0 - alpha * q
#         k += 1
#     end = time.time()
#     duree = end - start
#
#     return x, k, duree
