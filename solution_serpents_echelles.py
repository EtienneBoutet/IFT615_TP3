# -*- coding: utf-8 -*-

#####
# Étienne Boutet - boue2327
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
        B[s] = mdp.recompenses[s] * -1
        A[s, s] = -1

        for t in mdp.modele_transition[(s, plan.get(s))]:
            A[s, t[0]] = mdp.escompte * t[1] + A[s, t[0]]

    mdp.assigner_plan(plan)
    
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
    plan = {}

    for e in mdp.etats:
        candidat = (-1, None) 

        for action in mdp.actions[e]:
            val = 0
            for t in mdp.modele_transition[(e, action)]:
                val += t[1] * valeur[t[0]]
            
            # Trouver l'action maximum
            if val > candidat[0]:
                candidat = (val, action)

        # Comparer l'action maximum avec la valeur présente
        if candidat[0] > valeur[e]:
            plan[e] = candidat[1]
        else:
            plan[e] = mdp.plan.get(e)

    # Dernier élément du plan est toujours '1'.
    plan[mdp.etats[-1]] = '1'

    return plan

#####
# iteration_politiques: Algorithme d'itération par politiques, qui retourne le plan optimal et sa valeur.
#
# plan_initial: Le plan à utiliser pour initialiser l'algorithme d'itération par politiques.
#
# retour: Un tuple contenant le plan optimal et son tableau de valeurs.
### 
def iteration_politiques(mdp, plan_prime):
    plan = None

    while plan != plan_prime:
        plan = plan_prime
        valeur = calcul_valeur(mdp, plan)
        plan_prime = calcul_plan(mdp, valeur)

    return plan_prime, valeur
