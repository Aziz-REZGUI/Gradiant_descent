import random
import time
from sys import exit
import numpy as np
import sympy as sp
from sympy import *
import os
import scipy.linalg as spl
import matplotlib.pyplot as plt
from numpy import linalg as LA, matrix, double
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



def clear():
    global x,y,f
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
           2 : f2 = None  
           3 : f3(x,y) = 5 * x ** 2 + 3 * y ** 2 + 2 * x * y"""
              )  # TODO write this function's message
        choix = input("\nEntrez votre choix [1-3] : ")
        global func, x, y, A, B
        if choix == '1':
            func = (1 - x) ** 2 + 100 * (y - (x ** 2)) ** 2
            # func = pow(1-x,2)+100*pow(y-pow(x,2),2)
            global rosen
            rosen = true
            break
        elif choix == '2':
            global A, B, Matrix
            A = np.array([[3, -1, 0, 0, 0],
                          [-1, 12, -1, 0, 0],
                          [0, -1, 24, -1, 0],
                          [0, 0, -1, 48, -1],
                          [0, 0, 0, -1, 96]])
            B = np.array([[1], [2], [3], [4], [5]])
            Matrix = true
            x, y = symbols('x y', real=True)
            x = np.zeros((len(A), 1))
            # x = matrix[x]

            func = fn(x, A, B, 0)
            # TODO how to draw this ?? abir ??
            break

        elif choix == '3':
            func = 5 * x ** 2 + 3 * y ** 2 + 2 * x * y  # TODO to change with quadratic polynomial : Manel
            A = hessienne(func)
            B = conv_B(func)
            break
        else:
            print("choix incorrecte")
            functions()
            break


def fn(x, A, b, c=0.0):
    return 0.5 * np.transpose(x) @ A @ x - np.transpose(b) @ x + c


def entree():
    # instance les variables pour le fonctionnement de eval

    global string_func

    a, b, c = [int(x) for x in input("""Votre fonction est sous la forme f(x,y)=ax**2+by**2+cxy:
                            sachant que x**2 est x au carrée \nentez les coeff a,b,c séparées par virgule',' """).split(
        ',')]

    global func, x, y, rosen, A, B

    func = a * x ** 2 + b * y ** 2 + c * x * y
    rosen = false
    A = hessienne(func)
    B = conv_B(func)
    # TODO  force user to enter quadratic form


def entrer_matrice():
    n = int(input("Entrer le dimension de la matrice"))
    global A
    global B
    global eps, Matrix
    Matrix = true
    A = np.zeros((n, n))
    B = np.zeros((n, 1))
    for r in range(0, n):
        for c in range(0, n):
            A[(r), (c)] = (input("Element a[" + str(r + 1) + "," + str(c + 1) + "] "))

    for i in range(0, n):
        B[(i)] = (input('b[' + str(i + 1) + ']: '))


# TODO conversion de A et B : Manel

def choix_entree():
    """ Affiche le meneu d choix de saisie """
    while True:
        # output.clear()

        print("Choisissez l'option que vous voulez utilisé [1-2]: ")
        print("""
               1 : saisir une fonction polynomiale quadratique (f(x,y)=ax**2+by**2+cxy).
               2 : Saisir une fonction matricielle.
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
            main()
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


def graph_niv(func):
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
    # ax.contour(X, Y, Z, 10, colors="k", linestyles="solid")
    ax.view_init(20, 70)
    plt.show()


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
    ax.set_ylabel('y (cm)')
    plt.show()


def grad(f):
    x, y = symbols('x y', real=True)
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


def Rozen_brock_GC():
    x, y = symbols('x y', real=True)
    f = func
    print("la fonction est : ", f)
    print("solution de rozen brook avec la methode de gradient conjugé : \n ")
    f = lambdify([(x, y)], func, "numpy")
    x0 = [random.randint(0, 10), random.randint(0, 10)]
    x = x0[0]
    y = x0[1]
    print(op.fmin_cg(f, (x0[0], x0[1])))
    print("\n")


def conv_B(f):
    xx = diff(f, x)
    print(xx)
    yy = diff(f, y)
    print(yy)
    imageX = lambdify([x, y], xx)
    imageY = lambdify([x, y], yy)
    tab = [imageX(0, 0), imageY(0, 0)]
    return tab


def comparatif(A, B):
    tol = 1e-5  # La précision fixée à 10e-5
    if rosen:
        x, y = sp.symbols('x y', real=True)
        f = lambdify([(x, y)], func, "numpy")
        list = [[0, 0], [1, 2], [2, 3]]  # TODO to check with omar
        for i in [0, 1, 2]:
            # z = f(x, y)
            # x0 = random.choice(list)
            x0 = [random.randint(0, 10), random.randint(0, 10)]
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


def graph_Mat(A, b, c):
    fig = plt.figure(figsize=(10, 8))
    qf = fig.add_subplot(projection='3d')
    size = 100
    x1 = list(np.linspace(-6, 6, len(A)))
    x2 = list(np.linspace(-6, 6, len(A)))
    x1, x2 = np.meshgrid(x1, x2)
    zs = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            x = np.matrix([[x1[i, j]], [x2[i, j]]])
            zs[i, j] = fn(x, A, b, c)
    qf.plot_surface(x1, x2, zs, rstride=1, cstride=1, linewidth=0)
    fig.show()
    return x1, x2, zs


def niveau4(x1, x2, zs, steps=None):
    fig = plt.figure(figsize=(6, 6))
    cp = plt.contour(x1, x2, zs, 10)
    plt.clabel(cp, inline=1, fontsize=10)
    if steps is not None:
        steps = np.matrix(steps)
        plt.plot(steps[:, 0], steps[:, 1], '-o')
    fig.show()


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
            if Matrix:
                graph_Mat(A, B, 0)
            else:
                graph(func)

            niveau_2()
        elif choix == '2':
            graph_niv(func)
            niveau_2()
        elif choix == '3':
            print("le gradiant est ")
            print(grad(func))
            niveau_2()
        elif choix == '4':
            print("l'hessienne est ")
            print(hessienne(func))
            niveau_2()
        elif choix == '5':
            niveau_3()
        elif choix == '6':
            comparatif(A, B)
            niveau_2()
        elif choix == '7':
            main()
            break
        else:
            print("choix incorrecte")
            niveau_2()
        # output.clear()
        #         # os.clear()
        exit()


def niveau_3():
    if rosen:
        Rozen_brock_GC()
        niveau_2()
    else:
        pas, eps, x0, y0 = 0, 0, 0, 0
        # os.system('clear')
        # output.clear()
        print("""saisir vos propres valeurs de eps,pas,vecteur X0 """)
        # Lecture des paramétres propre à l'utilisateur :
        # Lecture de eps :
        print("Saisir la precision eps: \n")
        while True:
            try:
                eps = np.double(input("eps doit etre sous la forme x.xxxx"))
                break
            except ValueError:
                print("erreur de saisie ")

        # Lecture de vecteur de départ :
        print("saisir X0 le vecteur de départ ")
        print("X0=[x y]")

        # pour X :
        while True:
            try:
                x0 = double(input("saisir x "))
                break
            except ValueError:
                print("erreur de saisie")

        # pour Y :
        while True:
            try:
                y0 = double(input("saisir y "))
                break
            except ValueError:
                print("erreur de saisie")

        print(" le vecteur de départ X0=[", x0, " ", y0, "]")

        # Lecture du pas :
        depart = [x0, y0]

        step(depart, eps)
        graph_Mat()


def step(x0, eps):
    while True:
        # os.system('clear')
        # output.clear()
        print("Choisissez l'option que vous voulez utilisé [1-2]: ")
        print("""
            1 : voulez vous utiliser un pas fixe?
            2 : voulez vous utiliser un pas optimal(variable)"""
              )
        choix = input("\nEntrez votre choix [1-2] : \n")
        if choix == 1:
            print("saisir le pas(alpha) :\n ")
            while True:
                try:
                    pas = np.double(input("pas(alpha) doit etre different de 0 "))
                    break

                except ValueError:
                    print("erreur de saisie")
            x, k, du = conjugue(A, B, x0, 100, eps, pas)
            print("le mininum est ", x)
            # niveau_2()
            x1, x2, zs = graph_Mat(A, B, 0)
            niveau4(x1, x2, zs)

        elif choix == '2':
            x, k, du = conjugue(A, B, x0, 100, eps)
            print("le mininum est ", x)
            x1, x2, zs = graph_Mat(A, B, 0)
            niveau4(x1, x2, zs)
            # niveau_2()
        else:
            print("choix incorrecte")
            step(x0, eps)
            # output.clear()
            #         # os.clear()
            exit()


# TODO

if __name__ == '__main__':
    main()
