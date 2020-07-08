# -*- coding: utf-8 -*-

#####
# VotreNom (VotreMatricule) .~= À MODIFIER =~.
###

from pdb import set_trace as dbg  # Utiliser dbg() pour faire un break dans votre code.

import numpy as np

#################################
# Solution serpents et échelles #
#################################

#####
# calcul_valeur: Fonction qui retourne le tableau de valeurs d'un plan (politique).
#
# mdp: Spécification du processus de décision markovien (objet de la classe SerpentsEchelles, héritant de MDP).
#
# plan: Un plan donnant l'action associée à chaque état possible (dictionnaire).
#
# retour: Un tableau Numpy 1D de float donnant la valeur de chaque état du mdp, selon leur ordre dans mdp.etats.
### 
def calcul_valeur(mdp, plan):
    K = len(mdp.etats)

    A = np.zeros((K, K))
    B = np.zeros(K, )

    for s in plan:
        action = plan.get(s)

        transitions = mdp.modele_transition[(s, action)]

        B[s] = mdp.recompenses[s] * -1
        A[s, s] = -1

        for t in transitions:
            A[s, t[0]] = mdp.escompte * t[1] + A[s, t[0]]

    Ainv = np.linalg.inv(A)

    return np.dot(Ainv, B)


#####
# calcul_plan: Fonction qui retourne un plan à partir d'un tableau de valeurs.
#
# mdp: Spécification du processus de décision markovien (objet de la classe SerpentsEchelles, héritant de MDP).
#
# valeur: Un tableau de valeurs pour chaque état (tableau Numpy 1D de float).
#
# retour: Un plan (dictionnaire) qui maximise la valeur future espérée, en fonction du tableau "valeur".
### 
def calcul_plan(mdp, valeur):
    #TODO: .~= À COMPLÉTER =~.
    return dict( [ (s,mdp.actions[s][0]) for s in mdp.etats] )

#####
# iteration_politiques: Algorithme d'itération par politiques, qui retourne le plan optimal et sa valeur.
#
# plan_initial: Le plan à utiliser pour initialiser l'algorithme d'itération par politiques.
#
# retour: Un tuple contenant le plan optimal et son tableau de valeurs.
### 
def iteration_politiques(mdp, plan_prime):
    plan = None

    while plan is not plan_prime:
        plan = plan_prime
        valeur = calcul_valeur(mdp, plan)
        plan_prime = calcul_plan(mdp, valeur)

    return plan, None
