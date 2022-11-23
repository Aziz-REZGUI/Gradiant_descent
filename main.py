# PROJET 4: Notre projet permet de tracer une courbe en 3D , les lignes de niveaux, calculer le vecteur gradient
# conjugué standard par un pas fixe ou optimal et calculer la matrice Hessienne
# realisé par : Aziz REZGUI & ABIR BEN ABID & EYA SKOURI & SOUHA KHADRANI & OMAR BOUACHIR & MANEL FITOURI

# Implémentation des bibliothéques nécessaires
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

# déclarations des variables
x, y = symbols('x y', real=True)
func = None
f1 = None
string_func = ""
A = None
B = None
eps = None
Matrix = false
rosen = false
tras = false
memoryMatrix = false
entrerMatrice = false


def clear():
    global x, y, f, func, f1, string_func, A, B, eps, Matrix, rosen, tras, memoryMatrix, entrerMatrice
    x, y = symbols('x y', real=True)
    func = None
    f1 = None
    string_func = ""
    A = None
    B = None
    eps = None
    Matrix = false
    rosen = false
    tras = false
    memoryMatrix = false
    entrerMatrice = false


def enregistrer(f, X, nbI):
    try:
        file = open('resultat.txt', 'w2')
        file.write(f"* La fonction est : {f}\n")
        file.write(f"* La solution (xmin) = {X[0]}\n")
        file.write(f"* Valeur minimal de f (fmin) = {X[1]}\n")
        file.write(f"* nomb1re d'itiration  = {nbI}\n")
        file.write(f"---------------------------------------------------\n")
        file.close()
        print("Le résultat est enregistré avec succès dans 'Resultats.txt'")
    except Exception as e:
        print("Le résultat n'a pas été enregistré")


# menu qui offre le choix soit d'utiliser une fonction de la memoire ou de saisir sa propre fonction
def main():
    """ Affiche le menu principale du programme qui contient le choix de l'utilisateur"""
    while True:

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
        exit()


# menu pour choisir la fonction de la mémoire à utiliser
def functions():
    """ qu'elle fonction voulez vous choisir? """
    while True:
        os.system('clear')
        print("Choisissez l'option que vous voulez utilisé [1-3]: ")
        print(
            "--------\n1.𝑓(𝑥, 𝑦) =(1 − 𝑥) ² + 100(𝑦 − 𝑥 ²)²\n2.𝑓(𝑥, 𝑦,) =1/2〈𝐴𝑥, 𝑥〉 − 〈𝑏, 𝑥〉 \n3.𝑓(𝑥, 𝑦) =5𝑥 ² + 3y² +2xy \n-----------------\n")

        choix = input("\nEntrez votre choix [1-3] : ")
        global func, x, y, A, B
        if choix == '1':
            func = (1 - x) ** 2 + 100 * (y - (x ** 2)) ** 2
            global rosen
            rosen = true
            break
        elif choix == '2':
            global A, B, Matrix, memoryMatrix
            # saisie de la matrice A
            A = np.array([[3, -1, 0, 0, 0],
                          [-1, 12, -1, 0, 0],
                          [0, -1, 24, -1, 0],
                          [0, 0, -1, 48, -1],
                          [0, 0, 0, -1, 96]])
            # saisie de la matrice b
            B = np.array([[1], [2], [3], [4], [5]])
            Matrix = true
            memoryMatrix = true
            x, y = symbols('x y', real=True)
            x = np.zeros((len(A), 1))
            func = fn(x, A, B, 0)  # la fct fn permet d'entrer la 2eme fct
            break

        elif choix == '3':
            func = 5 * x ** 2 + 3 * y ** 2 + 2 * x * y
            A = np.array(hessienne(func), dtype=float)  # recupération de la matrice A à partir de l'hessinne
            B = np.array(conv_B(func), dtype=float)  # utilisation de la fonction con_B pour générer b
            break
        else:
            print("choix incorrecte")
            functions()
            break


# fonction qui permet de saisir la 2eme fonction
def fn(x, A, b, c=0.0):
    return 0.5 * np.transpose(x) @ A @ x - np.transpose(b) @ x + c


# menu qui permet de choisir la nature de la fonction entrée par l'utilisateur
def choix_entree():
    """ Affiche le meneu d choix de saisie """
    while True:

        print("Choisissez l'option que vous voulez utilisé [1-2]: ")
        print("""
               1 : saisir une fonction polynomiale quadratique (f(x,y)=ax**2+by**2+cxy).
               2 : Saisir une fonction matricielle.
               3 : revenir au menu précédant.""")

        choix = input("\nEntrez votre choix [1-3] : ")
        if choix == '1':
            entree()
            niveau_2()
        if choix == '2':
            entrer_matrice()
            niveau_2()
        if choix == '3':
            main()
            break
        else:
            print("choix incorrecte")
            choix_entree()
        exit()


# fonction qui permet de récupérer une fonction écrite par l'utilisateur en donnant les coeff a,b,c tout en acceptant la forme exigée
def entree():
    # instance les variables pour le fonctionnement de eval
    global string_func

    a, b, c = [int(x) for x in input("""Votre fonction est sous la forme f(x,y)=ax**2+by**2+cxy:
                            sachant que x**2 est x au carrée \nentez les coeff a,b,c séparées par virgule',' """).split(
        ',')]

    global func, x, y, rosen, A, B
    func = a * x ** 2 + b * y ** 2 + c * x * y
    rosen = false
    # recupération de la matrice A à partir de l'hessinne
    A = np.array(hessienne(func), dtype=float)
    # utilisation de la fonction con_B pour générer b
    B = np.array(conv_B(func), dtype=float)


# fonction qui permet de déterminer la matrice b (c'est a dire grand(f(x==0,Y==0))) à partir de la fonction polynome ecrite par l'utilisateur
def conv_B(f):
    xx = diff(f, x)
    print(xx)
    yy = diff(f, y)
    print(yy)
    imageX = lambdify([x, y], xx)
    imageY = lambdify([x, y], yy)
    tab = [imageX(0, 0), imageY(0, 0)]
    return tab


#
# fonction qui permet de récupérer une matrice écrite par l'utilisateur en donnant la dimension
def entrer_matrice():
    print("votre matrice est de taille")
    n = 2
    global A
    global B
    global eps, Matrix,tras
    global entrerMatrice
    entrerMatrice = true
    Matrix = true
    tras=true
    A = np.zeros((n, n))
    B = np.zeros((n, 1))
    for r in range(0, n):
        for c in range(0, n):
            A[(r), (c)] = (input("Element a[" + str(r + 1) + "," + str(c + 1) + "] "))

    for i in range(0, n):
        B[(i)] = (input('b[' + str(i + 1) + ']: '))


def convAToFunc(A):
    x, y = sp.symbols('x y', real=True)
    mul = [x, y]
    f = A[0][0] * mul[0] + A[1][0] * mul[0] + A[0][1] * mul[1] + A[1][1] * mul[1]
    return f


# menu qui offre à l'utilisateur les différents fontions offerte par le programme
def niveau_2():
    """ qu'elle traitement voulez vous effectuer?"""
    while True:
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
            if Matrix and tras:
                graph_Mat(A, B, 0)
            elif not Matrix and not tras:
                graph(func)
            elif Matrix and not tras:
                print("on peut pas dessiner cette fonction\n")
            niveau_2()
        elif choix == '2':
            if Matrix:
                print("on ne peut pas")
                niveau_2()
            graph_niv(func)
            niveau_2()
        elif choix == '3':
            if Matrix and len(A)<3:
                print("on ne peut pas calculer le gradient d'une matrice de taille 5*5")
                niveau_2()
            else:
                print("le gradiant est ")
                print(grad(func))
                niveau_2()
        elif choix == '4':
            if Matrix :
                print("on ne peut pas calculer la matrice hessienne d'une fonction matriciel avec taille de A>2")
                niveau_2()
            elif Matrix and entrerMatrice:
                print("l'hessienne est ... ")
                h = convAToFunc(A)
                print(hessienne(h))
                niveau_2()
            else:
                print("l'hessienne est ")
                print(hessienne(func))
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
        exit()


# Tracage de courbe en 3D :
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


# tracage de graphe + lignes de niveaux
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
    ax.view_init(20, 70)
    plt.show()


# tracage d'une matrice
def graph_Mat(A, b, c):
    fig = plt.figure(figsize=(10, 8))
    qf = fig.add_subplot(projection='3d')
    size = 200
    x1 = list(np.linspace(-6, 6, size))
    x2 = list(np.linspace(-6, 6, size))
    x1, x2 = np.meshgrid(x1, x2)
    zs = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            x = np.matrix([[x1[i, j]], [x2[i, j]]])
            zs[i, j] = fn(x, A, b, c)
    qf.plot_surface(x1, x2, zs, rstride=1, cstride=1, linewidth=0)
    fig.show()
    return x1, x2, zs


class bcolors:  # Affichage avec couleurs.
    OK = '\033[92m'  # vert
    RG = '\033[91m'  # rouge
    RESET = '\033[0m'  # rest des couleurs


# fonction qui permet de retourner le vecteur gradient d'une fonction f (le gradient d'une fct est la dérivée premier par rapport à chaque composante (x,y..))
def grad(f):
    x, y = symbols('x y', real=True)
    tab = [diff(f, x), diff(f, y)]
    return tab


# fonction qui permet de retourner le vecteur gradient d'une fonction f (la matrice hessienne est la matrice carrée de ses dérivées partielles secondes )
def hessienne(f):
    params = [x, y]
    gradd = grad(f)
    tab = np.array([[None] * 2, [None] * 2])
    for i in range(2):
        for j in range(2):
            tab[i, j] = diff(gradd[i], params[j])
    return tab


# fonction qui permet de retourner le gradient conjugué standard d'une fonction de la forme Ax=b
# La méthode du gradient conjugué permet de résoudre les systémes linéaires dont la matrice est symétrique définie positive
# Il s’agit d’une méthode qui consiste à minimiser une fonction
def conjugue(A, b, X, itMax, tol, pas=0):
    steps = [(-2.0, -2.0)]
    if not is_pos_def(A):
        raise ValueError("\n A n'est pas symétrique définie positive")
    else:
        R = b - A @ X  # -gradient de f(Xk)
        P = R  # Direction initiale (-gradient de la fonction)
        k = 0
        alpha = pas
        if len(X) == 1:
            steps = [(-2.0)]
        elif len(X) == 2:
            steps = [(-2.0, -2.0)]
        start = time.time()
        while (k <= itMax) and (LA.norm(R) > tol):  # verification des condition: #La précision fixée à 10e-5
            # && nombre d'itération ne dépasse pas nbr max
            Ap = A.dot(P)  # A * P
            if not pas == 0:
                alpha = np.transpose(R).dot(R) / np.transpose(P).dot(Ap)  # pas
            X = X + (alpha * P)  # X(k+1) = X(k) + direction(k) * pas(k)
            if len(X) == 1:
                steps.append((X[0], X[1]))
            elif len(X) == 2:
                steps.append((X[0, 0], X[1, 0]))

            Rancien = R  # R(k) -->gradient f(k+1)
            R = R - (alpha * Ap)  # R(k+1) --> -gradient f(k+1)
            beta = np.transpose(R).dot(R) / np.transpose(Rancien).dot(Rancien)
            P = R + beta * P  # direction k+1

            k = k + 1  # incrémentation d'itération
        end = time.time()
        duree = end - start
        print("\nle nombre d'itération = \n", k)

        print("\n notre solution minimale cherchée X = \n", X)

        # return result , nb it , exec duration
        return X, k, duree, steps


# fonction qui permet de savoir si une matrice est symétrique positive ou non
def is_pos_def(x):
    """symetrique positive ou nn"""
    return (np.array_equal(A, np.transpose(A))) and (np.all(np.linalg.eigvals(A) > 0)) and (spl.det(A) != 0)


# fonction qui permet de calculer le minimum de la fct rosenbrock
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


# fonction qui permet de donner le comparatif avec 3 X0 différents
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
            X, k, du, s = conjugue(A, B, x0, 100, 1.5e-8)
            print("la resultat n°\n", i, "=", X, "avec un vecteur de depart x0 :", x0, "et une durée d'exec = ", du,
                  "et un nbr exec ", k)


# menu qui permet de recupérer de l'utilisateur les valeurs nécessaires à l'éxection de la méthode du gradient conjugé
# ( la précision eps, X0 (le vecteur de départ) et le pas de la méthode)
def niveau_3():
    if rosen:
        Rozen_brock_GC()
        niveau_2()
    else:
        pas, eps, x0, y0 = 0, 0, 0, 0  # initialisation
        print("""saisir vos propres valeurs de eps,pas,vecteur X0 """)
        # Lecture des paramétres propre à l'utilisateur
        # Lecture de eps(la précision)
        print("Saisir la precision eps: \n")
        while True:
            try:
                eps = np.double(input("eps doit etre sous la forme x.xxxx"))
                break
            except ValueError:
                print("erreur de saisie ")

        # Lecture de vecteur de départ (X0)
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

        # Lecture du pas de la méthode
        depart = [x0, y0]

        step(depart, eps)
        graph_Mat()


# menu qui permet de choisir le pas (pas fixe ou pas optimal)
def step(x0, eps):
    while True:
        # du = None
        print("Choisissez l'option que vous voulez utilisé [1-2]: ")
        print("""
            1 : voulez vous utiliser un pas fixe
            2 : voulez vous utiliser un pas optimal"""
              )
        choix = input("\nEntrez votre choix [1-2] : \n")
        if choix == '1':
            print("saisir le pas(alpha) :\n ")
            while True:
                try:
                    pas = np.double(input("pas(alpha) doit etre different de 0 "))
                    break

                except ValueError:
                    print("erreur de saisie")
            x0, k, du, s = conjugue(A, B, x0, 100, eps, pas)
            enregistrer(func, x, du)
            print("le mininum est ", x0)
            # niveau_2()
            x1, x2, zs = graph_Mat(A, B, 0)
            niveau4(x1, x2, zs, s)

        elif choix == '2':

            x0, k, du, s = conjugue(A, B, x0, 100, eps)
            print("le mininum est ", x0)
            x1, x2, zs = graph_Mat(A, B, 0)
            niveau4(x1, x2, zs, s)
            # niveau_2()
        else:
            print("choix incorrecte")
            step(x0, eps)
            # output.clear()
            #         # os.clear()
            exit()


def niveau4(x1, x2, zs, steps=None):
    fig = plt.figure(figsize=(6, 6))
    cp = plt.contour(x1, x2, zs, 10)
    # X, k, duree, steps = conjugue(A, B, x, 100, 1e-10)
    plt.clabel(cp, inline=1, fontsize=10)
    if steps is not None:
        steps = np.matrix(steps)
        plt.plot(steps[:, 0], steps[:, 1], '-o')
    fig.show()


if __name__ == '__main__':
    main()
