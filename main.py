from sys import exit
import numpy as np
import sympy as sp
from sympy import *
import os
import matplotlib.pyplot as plt

# jarrabt el hessienne wel grad yekhdmou maa l graph wel l de niveau jawna ahla jaw
x, y = symbols('x y', real=True)
func = None
f1 = None
string_func = ""


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
        global func
        if choix == '1':
            func = (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
            break
        elif choix == '2':
            break
            # func = None #TODO ken araft kifet tdakhl la 2eme fction just hottha houni MANEL/EYA
        elif choix == '3':
            func = x * exp(-x ** 2 - y ** 2)
            break
        else:
            print("choix incorrecte")
            functions()
            break

        # os.clear()
        # output.clear()
        # exit()


def entree():
    # instance les variables pour le fonctionnement de eval
    # nbvariable = int(input("Donner le nombre de variables"))
    global string_func
    string_func = input("Entrer votre fonction avec des parametres x et y  :").lower()
    # X = extract_symbols(string_func, nbvariable)
    # fct = lambda x, y: string_func
    global func
    func = eval(string_func)
    print(func)


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
            entree()
            niveau_2()
        elif choix == '3':
            break
        else:
            print("choix incorrecte")
            main()
        os.clear()

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
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=5, cstride=5,
                    cmap='cool')
    ax.set_title("Surface Bonus", fontsize=13)
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('y', fontsize=11)
    ax.set_zlabel('Z', fontsize=11)
    plt.show()


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


def grad():
    # x,y=symbols('x y',real=True)
    f = func
    tab = [diff(f, x), diff(f, y)]
    return tab


def hessienne():
    params = [x, y]
    gradd = grad()
    tab = np.array([[None] * 2, [None] * 2])
    for i in range(2):
        for j in range(2):
            tab[i, j] = diff(gradd[i], params[j])
    return tab


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
        choix = input("\nEntrez votre choix [1-7] : ")
        if choix == '1':
            graph(func)
        elif choix == '2':
            graph_niveau(func)
        elif choix == '3':
            print("le gradiant est ")
            print(grad())
        elif choix == '4':
            print("l'hessienne est ")
            print(hessienne())
        # elif choix == '5':
        #     conjugue(A, b, X, itMax, tol)
        # elif choix == '6':
        #     #######
        elif choix == '7':
            entree()
            break
        else:
            print("choix incorrecte")
            niveau_2()
        # output.clear()
        # os.clear()
        exit()


if __name__ == '__main__':
    main()
# def menu_fonctions()
