# -*- coding: utf-8 -*-

#####
# Étienne Boutet - boue2327
# Raphael Valois - valr2802
###

from pdb import set_trace as dbg  # Utiliser dbg() pour faire un break dans votre code.

import numpy as np
import operator

class Correcteur:
    def __init__(self, p_init, p_transition, p_observation, int2letters, letters2int):
        '''Correcteur de frappes dans un mot.

        Modèle de Markov caché (HMM) permettant de corriger les erreurs de frappes
        dans un mot. La correction est dictée par l'inférence de l'explication
        la plus pausible du modèle.

        Parameters
        ------------
        p_init : array-like shape (N,)
                 Probabilités initiales de l'état caché à la première position.

        p_transition : array-like shape (X,Y)
                       Modèle de transition.

        p_observation : array-like shape (X,Y)
                        Modèle d'observation.

        int2letters : list
                      Associe un entier (l'indice) à une lettre.

        letters2int : dict
                      Associe une lettre (clé) à un entier (valeur).
        '''
        self.p_init = p_init
        self.p_transition = p_transition
        self.p_observation = p_observation
        self.int2letters = int2letters
        self.letters2int = letters2int

    def corrige(self, mot):
        '''Corrige les frappes dans un mot.

        Retourne la correction du mot donné et la probabilité p(mot, mot corrigé).

        Parameters
        ------------
        mot : string
              Mot à corriger.

        Returns
        -----------
        mot_corrige : string
                      Le mot corrigé.

        prob : float
               Probabilité dans le HMM du mot observé et du mot corrigé.
               C'est-à-dire 'p(mot, mot_corrige)'.
        '''

        T = len(mot)
        K = self.p_transition.shape[0]
        alpha = np.empty((K, T))
        beta = np.empty((K, T))

        # Création des tables alpha et beta
        alpha[:, 0] = self.p_init * self.p_observation[self.letters2int[mot[0]], :]
        beta[:, 0] = 0

        for t in range(1, T):
            for i in range(K):
                alpha[i, t] = self.p_observation[self.letters2int[mot[t]], i] * np.max(self.p_transition[i, :] * alpha[:, t - 1])
                beta[i, t] = np.argmax(alpha[:, t - 1] * self.p_transition[i, :])

        # Construction du mot corrigé
        result = ""
        ind = np.argmax(alpha[:, T - 1])
        result += self.int2letters[int(ind)]

        for j in range(T - 1, 0, -1):
            ind = beta[int(ind), j]
            result += self.int2letters[int(ind)]

        result = result[::-1]

        # Calcul de la probabilité de p(mot, mot_corrige)
        prob = self.p_init[self.letters2int[result[0]]] * self.p_observation[self.letters2int[mot[0]], self.letters2int[result[0]]]

        for j in range(1, len(mot)):
            prob *= self.p_transition[self.letters2int[result[j]], self.letters2int[result[j - 1]]] * self.p_observation[self.letters2int[mot[j]], self.letters2int[result[j]]]

        return result, prob
