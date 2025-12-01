"""
Distillation de Mélanges Multicomposants
==========================================
Modélisation et simulation complète avec méthodes simplifiées et rigoureuses

Auteur: Prof. BAKHER Zine Elabidine
Cours: Modélisation et Simulation des Procédés - PIC
Université uh1
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, brentq, minimize
from scipy.linalg import solve_banded
from thermo.chemical import Chemical
from thermo import ChemicalConstantsPackage, PRMIX, CEOSLiquid, CEOSGas
import warnings
warnings.filterwarnings('ignore')

# Pour visualisations interactives
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠ Plotly non disponible. Installation: pip install plotly")
    print("  Les visualisations interactives ne seront pas disponibles.")

class Compound:
    """
    Représente un composé chimique avec ses propriétés thermodynamiques
    """
    
    def __init__(self, name):
        """
        Initialise le composé depuis la base de données thermo
        
        Parameters:
        -----------
        name : str
            Nom du composé (ex: 'benzene', 'toluene', 'xylene')
        """
        self.name = name
        try:
            self.chem = Chemical(name)
            
            # Propriétés critiques
            self.Tc = self.chem.Tc  # Température critique (K)
            self.Pc = self.chem.Pc  # Pression critique (Pa)
            self.omega = self.chem.omega  # Facteur acentrique
            
            # Propriétés normales
            self.Tb = self.chem.Tb  # Température d'ébullition normale (K)
            self.MW = self.chem.MW  # Masse molaire (g/mol)
            
            # Pour l'enthalpie
            self.Hfus = self.chem.Hfusm if self.chem.Hfusm else 0  # Enthalpie de fusion
            
        except Exception as e:
            raise ValueError(f"Impossible de charger le composé '{name}': {e}")
    
    def vapor_pressure(self, T):
        """
        Calcule la pression de vapeur saturante à la température T
        
        Parameters:
        -----------
        T : float
            Température (K)
        
        Returns:
        --------
        Psat : float
            Pression de vapeur saturante (Pa)
        """
        self.chem.T = T
        Psat = self.chem.Psat
        return Psat if Psat else 1e-10  # Éviter division par zéro
    
    def K_value(self, T, P):
        """
        Calcule le coefficient de partage K = y/x
        
        Pour un mélange idéal: K = Psat(T) / P
        
        Parameters:
        -----------
        T : float
            Température (K)
        P : float
            Pression (Pa)
        
        Returns:
        --------
        K : float
            Coefficient de partage
        """
        Psat = self.vapor_pressure(T)
        return Psat / P
    
    def enthalpy_liquid(self, T, T_ref=298.15):
        """
        Calcule l'enthalpie du liquide à T par rapport à T_ref
        
        Parameters:
        -----------
        T : float
            Température (K)
        T_ref : float
            Température de référence (K)
        
        Returns:
        --------
        H_L : float
            Enthalpie molaire liquide (J/mol)
        """
        self.chem.T = T
        try:
            # Utiliser la capacité calorifique pour l'intégration
            Cp_liquid = self.chem.Cplm  # J/(mol·K)
            H_L = Cp_liquid * (T - T_ref)
            return H_L
        except:
            # Approximation simple si Cp n'est pas disponible
            return 4.18 * self.MW * (T - T_ref)  # Approximation eau
    
    def enthalpy_vapor(self, T, T_ref=298.15):
        """
        Calcule l'enthalpie de la vapeur à T par rapport à T_ref
        
        Parameters:
        -----------
        T : float
            Température (K)
        T_ref : float
            Température de référence (K)
        
        Returns:
        --------
        H_V : float
            Enthalpie molaire vapeur (J/mol)
        """
        H_L = self.enthalpy_liquid(T, T_ref)
        self.chem.T = T
        Hvap = self.chem.Hvap if self.chem.Hvap else 40000  # J/mol (valeur typique)
        return H_L + Hvap
    
    def __repr__(self):
        return f"Compound(name='{self.name}', Tb={self.Tb-273.15:.1f}°C, MW={self.MW:.2f})"


class ThermodynamicPackage:
    """
    Package thermodynamique pour calculs d'équilibre et propriétés de mélanges
    """
    
    def __init__(self, compounds):
        """
        Parameters:
        -----------
        compounds : list of Compound objects
            Liste des composés du mélange
        """
        self.compounds = compounds
        self.n_comp = len(compounds)
        self.compound_names = [c.name for c in compounds]
        
    def K_values(self, T, P, x=None):
        """
        Calcule tous les coefficients K à T et P
        
        Parameters:
        -----------
        T : float
            Température (K)
        P : float
            Pression (Pa)
        x : array, optional
            Compositions liquides (pour modèles non-idéaux)
        
        Returns:
        --------
        K : ndarray
            Coefficients K pour tous les composés
        """
        K = np.array([comp.K_value(T, P) for comp in self.compounds])
        return K
    
    def relative_volatilities(self, T, P, ref_index=-1):
        """
        Calcule les volatilités relatives par rapport au composé de référence
        
        Parameters:
        -----------
        T : float
            Température (K)
        P : float
            Pression (Pa)
        ref_index : int
            Index du composé de référence (par défaut: le plus lourd)
        
        Returns:
        --------
        alpha : ndarray
            Volatilités relatives
        """
        K = self.K_values(T, P)
        K_ref = K[ref_index]
        alpha = K / K_ref
        return alpha
    
    def bubble_temperature(self, P, x, T_guess=None, tol=1e-6, max_iter=100):
        """
        Calcule la température de bulle pour une composition liquide donnée
        
        Résout: sum(K_i * x_i) = 1
        
        Parameters:
        -----------
        P : float
            Pression (Pa)
        x : array
            Composition liquide (fractions molaires)
        T_guess : float, optional
            Estimation initiale de température (K)
        
        Returns:
        --------
        T_bubble : float
            Température de bulle (K)
        y : array
            Composition vapeur à l'équilibre
        """
        x = np.array(x)
        
        if T_guess is None:
            # Estimation: moyenne pondérée des Tb
            T_guess = np.sum([x[i] * self.compounds[i].Tb for i in range(self.n_comp)])
        
        def equation(T):
            K = self.K_values(T, P)
            return np.sum(K * x) - 1.0
        
        try:
            T_bubble = fsolve(equation, T_guess, full_output=False)[0]
            K = self.K_values(T_bubble, P)
            y = K * x
            y = y / np.sum(y)  # Normalisation
            return T_bubble, y
        except:
            print(f"⚠ Convergence difficile pour bubble T avec x={x}")
            return T_guess, x.copy()
    
    def dew_temperature(self, P, y, T_guess=None, tol=1e-6, max_iter=100):
        """
        Calcule la température de rosée pour une composition vapeur donnée
        
        Résout: sum(y_i / K_i) = 1
        
        Parameters:
        -----------
        P : float
            Pression (Pa)
        y : array
            Composition vapeur (fractions molaires)
        T_guess : float, optional
            Estimation initiale de température (K)
        
        Returns:
        --------
        T_dew : float
            Température de rosée (K)
        x : array
            Composition liquide à l'équilibre
        """
        y = np.array(y)
        
        if T_guess is None:
            # Estimation: moyenne pondérée des Tb
            T_guess = np.sum([y[i] * self.compounds[i].Tb for i in range(self.n_comp)])
        
        def equation(T):
            K = self.K_values(T, P)
            return np.sum(y / K) - 1.0
        
        try:
            T_dew = fsolve(equation, T_guess, full_output=False)[0]
            K = self.K_values(T_dew, P)
            x = y / K
            x = x / np.sum(x)  # Normalisation
            return T_dew, x
        except:
            print(f"⚠ Convergence difficile pour dew T avec y={y}")
            return T_guess, y.copy()
    
    def mixture_enthalpy_liquid(self, T, x, T_ref=298.15):
        """
        Calcule l'enthalpie du mélange liquide
        
        Hypothèse: mélange idéal (pas d'enthalpie de mélange)
        
        Parameters:
        -----------
        T : float
            Température (K)
        x : array
            Composition molaire liquide
        T_ref : float
            Température de référence (K)
        
        Returns:
        --------
        H_L : float
            Enthalpie molaire du mélange liquide (J/mol)
        """
        H_L = np.sum([x[i] * self.compounds[i].enthalpy_liquid(T, T_ref) 
                     for i in range(self.n_comp)])
        return H_L
    
    def mixture_enthalpy_vapor(self, T, y, T_ref=298.15):
        """
        Calcule l'enthalpie du mélange vapeur
        
        Parameters:
        -----------
        T : float
            Température (K)
        y : array
            Composition molaire vapeur
        T_ref : float
            Température de référence (K)
        
        Returns:
        --------
        H_V : float
            Enthalpie molaire du mélange vapeur (J/mol)
        """
        H_V = np.sum([y[i] * self.compounds[i].enthalpy_vapor(T, T_ref) 
                     for i in range(self.n_comp)])
        return H_V
    
    def print_properties(self, T, P):
        """
        Affiche les propriétés à T et P
        """
        print(f"\n{'PROPRIÉTÉS À T={T-273.15:.1f}°C, P={P/1e5:.3f} bar':-^80}")
        print(f"{'Composé':<15} {'Tb (°C)':<12} {'Psat (kPa)':<15} {'K':<12} {'α':<12}")
        print("-" * 80)
        
        K = self.K_values(T, P)
        alpha = self.relative_volatilities(T, P)
        
        for i, comp in enumerate(self.compounds):
            Psat = comp.vapor_pressure(T)
            print(f"{comp.name:<15} {comp.Tb-273.15:<12.2f} {Psat/1000:<15.2f} "
                  f"{K[i]:<12.4f} {alpha[i]:<12.4f}")
        print("-" * 80)


class ShortcutDistillation:
    """
    Méthodes simplifiées de dimensionnement (Fenske, Underwood, Gilliland, Kirkbride)
    """
    
    def __init__(self, thermo_package, F, z_F, P=101325):
        """
        Parameters:
        -----------
        thermo_package : ThermodynamicPackage
            Package thermodynamique
        F : float
            Débit d'alimentation (kmol/h)
        z_F : array
            Composition de l'alimentation (fractions molaires)
        P : float
            Pression de la colonne (Pa)
        """
        self.thermo = thermo_package
        self.F = F
        self.z_F = np.array(z_F)
        self.P = P
        self.n_comp = len(z_F)
        
        # Identifier les composés clés
        self._identify_key_components()
        
    def _identify_key_components(self):
        """
        Identifie les composés clés (léger et lourd)
        """
        # Volatilités relatives à une température moyenne
        T_avg = np.mean([comp.Tb for comp in self.thermo.compounds])
        alpha = self.thermo.relative_volatilities(T_avg, self.P)
        
        # Trier par volatilité décroissante
        sorted_indices = np.argsort(alpha)[::-1]
        
        # Clé léger: composé le plus volatil avec z_F significatif
        # Clé lourd: composé le moins volatil avec z_F significatif
        threshold = 0.01  # Seuil de composition significative
        
        for idx in sorted_indices:
            if self.z_F[idx] > threshold:
                self.LK_idx = idx  # Light Key
                break
        
        for idx in sorted_indices[::-1]:
            if self.z_F[idx] > threshold:
                self.HK_idx = idx  # Heavy Key
                break
        
        print(f"\n✓ Composés clés identifiés:")
        print(f"  Clé léger (LK): {self.thermo.compound_names[self.LK_idx]}")
        print(f"  Clé lourd (HK): {self.thermo.compound_names[self.HK_idx]}")
    
    def material_balance(self, recovery_LK_D=0.95, recovery_HK_B=0.95):
        """
        Calcule les bilans matières globaux
        
        Parameters:
        -----------
        recovery_LK_D : float
            Récupération du clé léger dans le distillat (fraction)
        recovery_HK_B : float
            Récupération du clé lourd dans le résidu (fraction)
        
        Returns:
        --------
        D : float
            Débit de distillat (kmol/h)
        B : float
            Débit de résidu (kmol/h)
        x_D : array
            Composition du distillat
        x_B : array
            Composition du résidu
        """
        # Débits des clés
        LK_in_feed = self.F * self.z_F[self.LK_idx]
        HK_in_feed = self.F * self.z_F[self.HK_idx]
        
        # Distribution selon les récupérations
        LK_in_D = recovery_LK_D * LK_in_feed
        LK_in_B = LK_in_feed - LK_in_D
        
        HK_in_B = recovery_HK_B * HK_in_feed
        HK_in_D = HK_in_feed - HK_in_B
        
        # Initialisation des compositions
        d = np.zeros(self.n_comp)  # Débits dans distillat
        b = np.zeros(self.n_comp)  # Débits dans résidu
        
        d[self.LK_idx] = LK_in_D
        d[self.HK_idx] = HK_in_D
        b[self.LK_idx] = LK_in_B
        b[self.HK_idx] = HK_in_B
        
        # Distribution des composés non-clés (approximation)
        for i in range(self.n_comp):
            if i != self.LK_idx and i != self.HK_idx:
                T_avg = np.mean([comp.Tb for comp in self.thermo.compounds])
                alpha = self.thermo.relative_volatilities(T_avg, self.P)
                
                if alpha[i] > alpha[self.LK_idx]:
                    # Plus léger que LK: tout dans distillat
                    d[i] = self.F * self.z_F[i]
                    b[i] = 0
                elif alpha[i] < alpha[self.HK_idx]:
                    # Plus lourd que HK: tout dans résidu
                    d[i] = 0
                    b[i] = self.F * self.z_F[i]
                else:
                    # Entre LK et HK: répartition proportionnelle
                    ratio = (alpha[i] - alpha[self.HK_idx]) / (alpha[self.LK_idx] - alpha[self.HK_idx])
                    d[i] = ratio * self.F * self.z_F[i]
                    b[i] = (1 - ratio) * self.F * self.z_F[i]
        
        D = np.sum(d)
        B = np.sum(b)
        
        x_D = d / D
        x_B = b / B
        
        self.D = D
        self.B = B
        self.x_D = x_D
        self.x_B = x_B
        
        return D, B, x_D, x_B
    
    def fenske_equation(self):
        """
        Calcule le nombre minimum de plateaux (reflux total)
        
        Équation de Fenske:
        N_min = log[(x_LK/x_HK)_D / (x_LK/x_HK)_B] / log(α_avg)
        
        Returns:
        --------
        N_min : float
            Nombre minimum de plateaux théoriques
        alpha_avg : float
            Volatilité relative moyenne
        """
        # Température moyenne entre tête et fond
        T_top = self.thermo.compounds[self.LK_idx].Tb
        T_bottom = self.thermo.compounds[self.HK_idx].Tb
        T_avg = (T_top + T_bottom) / 2
        
        # Volatilité relative moyenne
        alpha = self.thermo.relative_volatilities(T_avg, self.P)
        alpha_LK_HK = alpha[self.LK_idx] / alpha[self.HK_idx]
        
        # Rapports des clés
        ratio_D = self.x_D[self.LK_idx] / self.x_D[self.HK_idx]
        ratio_B = self.x_B[self.LK_idx] / self.x_B[self.HK_idx]
        
        # Nombre minimum de plateaux
        N_min = np.log(ratio_D / ratio_B) / np.log(alpha_LK_HK)
        
        self.N_min = N_min
        self.alpha_avg = alpha_LK_HK
        
        return N_min, alpha_LK_HK
    
    def underwood_method(self, q=1.0):
        """
        Calcule le reflux minimum par la méthode d'Underwood
        
        Parameters:
        -----------
        q : float
            Qualité de l'alimentation (1.0 = liquide saturé)
        
        Returns:
        --------
        R_min : float
            Rapport de reflux minimum
        theta : float
            Racine de l'équation d'Underwood
        """
        # Température moyenne
        T_avg = np.mean([comp.Tb for comp in self.thermo.compounds])
        alpha = self.thermo.relative_volatilities(T_avg, self.P)
        
        # Équation 1: Trouver theta
        def equation1(theta):
            return np.sum(alpha * self.z_F / (alpha - theta)) - (1 - q)
        
        # Theta est entre alpha_HK et alpha_LK
        alpha_HK = alpha[self.HK_idx]
        alpha_LK = alpha[self.LK_idx]
        
        try:
            theta = brentq(equation1, alpha_HK + 0.01, alpha_LK - 0.01)
        except:
            # Si brentq échoue, utiliser une valeur intermédiaire
            theta = (alpha_HK + alpha_LK) / 2
            print(f"⚠ Convergence difficile pour theta, utilisation de {theta:.3f}")
        
        # Équation 2: Calculer R_min
        R_min_plus_1 = np.sum(alpha * self.x_D / (alpha - theta))
        R_min = max(R_min_plus_1 - 1, 0.5)  # Minimum physique
        
        self.R_min = R_min
        self.theta = theta
        
        return R_min, theta
    
    def gilliland_correlation(self, R):
        """
        Calcule le nombre de plateaux avec la corrélation de Gilliland
        
        Parameters:
        -----------
        R : float
            Rapport de reflux opératoire
        
        Returns:
        --------
        N : float
            Nombre de plateaux théoriques
        """
        X = (R - self.R_min) / (R + 1)
        
        # Corrélation de Gilliland
        exponent = (1 + 54.4*X) * (X - 1) / ((11 + 117.2*X) * np.sqrt(X + 1e-10))
        Y = 1 - np.exp(exponent)
        
        # Nombre de plateaux
        N = self.N_min + Y / (1 - Y + 1e-10)
        
        return N
    
    def kirkbride_equation(self, N_total):
        """
        Détermine la position du plateau d'alimentation
        
        Parameters:
        -----------
        N_total : int
            Nombre total de plateaux
        
        Returns:
        --------
        N_R : int
            Nombre de plateaux en rectification
        N_S : int
            Nombre de plateaux en épuisement
        feed_stage : int
            Numéro du plateau d'alimentation
        """
        # Ratio selon Kirkbride
        ratio_term = (self.B / self.D) * \
                    (self.z_F[self.HK_idx] / self.z_F[self.LK_idx]) * \
                    (self.x_B[self.LK_idx] / self.x_D[self.HK_idx])**2
        
        log_ratio = 0.206 * np.log(ratio_term + 1e-10)
        N_R_over_N_S = np.exp(log_ratio)
        
        # Résolution
        N_S = N_total / (1 + N_R_over_N_S)
        N_R = N_total - N_S
        
        feed_stage = int(np.ceil(N_R)) + 1
        
        return int(np.ceil(N_R)), int(np.floor(N_S)), feed_stage
    
    def complete_shortcut_design(self, recovery_LK_D=0.95, recovery_HK_B=0.95,
                                 R_factor=1.3, q=1.0, efficiency=0.70):
        """
        Dimensionnement complet par méthodes simplifiées
        
        Parameters:
        -----------
        recovery_LK_D : float
            Récupération clé léger dans distillat
        recovery_HK_B : float
            Récupération clé lourd dans résidu
        R_factor : float
            Facteur multiplicatif pour reflux (R = R_factor * R_min)
        q : float
            Qualité alimentation
        efficiency : float
            Efficacité des plateaux
        
        Returns:
        --------
        results : dict
            Dictionnaire avec tous les résultats
        """
        print("\n" + "=" * 80)
        print("DIMENSIONNEMENT PAR MÉTHODES SIMPLIFIÉES")
        print("=" * 80)
        
        # 1. Bilans matières
        print("\n1. BILANS MATIÈRES GLOBAUX")
        D, B, x_D, x_B = self.material_balance(recovery_LK_D, recovery_HK_B)
        
        print(f"   Débit distillat: D = {D:.2f} kmol/h")
        print(f"   Débit résidu: B = {B:.2f} kmol/h")
        
        # 2. Fenske
        print("\n2. ÉQUATION DE FENSKE (reflux total)")
        N_min, alpha_avg = self.fenske_equation()
        print(f"   N_min = {N_min:.2f} plateaux théoriques")
        print(f"   α_avg(LK/HK) = {alpha_avg:.3f}")
        
        # 3. Underwood
        print("\n3. MÉTHODE D'UNDERWOOD (reflux minimum)")
        R_min, theta = self.underwood_method(q)
        print(f"   R_min = {R_min:.3f}")
        print(f"   θ = {theta:.3f}")
        
        # 4. Gilliland
        print("\n4. CORRÉLATION DE GILLILAND")
        R = R_factor * R_min
        N_theoretical = self.gilliland_correlation(R)
        print(f"   R opératoire = {R:.3f} ({R_factor}× R_min)")
        print(f"   N théorique = {N_theoretical:.2f} plateaux")
        
        # 5. Plateaux réels
        N_real = int(np.ceil(N_theoretical / efficiency))
        print(f"   Efficacité = {efficiency*100:.1f}%")
        print(f"   N réel = {N_real} plateaux")
        
        # 6. Kirkbride
        print("\n5. ÉQUATION DE KIRKBRIDE (position alimentation)")
        N_R, N_S, feed_stage = self.kirkbride_equation(N_real)
        print(f"   Plateaux rectification: {N_R}")
        print(f"   Plateaux épuisement: {N_S}")
        print(f"   Plateau d'alimentation: {feed_stage}")
        
        # 7. Débits internes
        print("\n6. DÉBITS INTERNES")
        L = R * D
        V = L + D
        L_prime = L + self.F * q  # Hypothèse: q fraction liquide
        V_prime = V
        
        print(f"   Liquide rectification: L = {L:.2f} kmol/h")
        print(f"   Vapeur rectification: V = {V:.2f} kmol/h")
        print(f"   Liquide épuisement: L' = {L_prime:.2f} kmol/h")
        print(f"   Vapeur épuisement: V' = {V_prime:.2f} kmol/h")
        
        print("\n" + "=" * 80)
        
        results = {
            'D': D,
            'B': B,
            'x_D': x_D,
            'x_B': x_B,
            'N_min': N_min,
            'alpha_avg': alpha_avg,
            'R_min': R_min,
            'theta': theta,
            'R': R,
            'N_theoretical': N_theoretical,
            'N_real': N_real,
            'efficiency': efficiency,
            'N_R': N_R,
            'N_S': N_S,
            'feed_stage': feed_stage,
            'L': L,
            'V': V,
            'L_prime': L_prime,
            'V_prime': V_prime,
            'recovery_LK_D': recovery_LK_D,
            'recovery_HK_B': recovery_HK_B
        }
        
        return results

# Suite dans le prochain fichier...