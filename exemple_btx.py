"""
Exemple Complet: Distillation du système BTX
==============================================
Séparation d'un mélange Benzène-Toluène-Xylène

Auteur: Prof. BAKHER Zine Elabidine
Cours: Modélisation et Simulation des Procédés - PIC
"""

import numpy as np
import sys
sys.path.append('.')

# Importer nos modules
from distillation_multicomposants import (
    Compound, ThermodynamicPackage, ShortcutDistillation
)
from visualization import DistillationVisualizer, print_design_summary

def exemple_btx_complet():
    """
    Exemple complet: Séparation BTX
    """
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + "DISTILLATION MULTICOMPOSANTS - SYSTÈME BTX".center(78) + "║")
    print("║" + "Benzène - Toluène - Xylène".center(78) + "║")
    print("╚" + "═" * 78 + "╝\n")
    
    # ========================================================================
    # 1. DÉFINITION DU SYSTÈME
    # ========================================================================
    print("1. DÉFINITION DU SYSTÈME")
    print("-" * 80)
    
    # Composés
    compound_names = ['benzene', 'toluene', 'o-xylene']
    print(f"   Composés: {', '.join(compound_names)}")
    
    compounds = []
    for name in compound_names:
        try:
            comp = Compound(name)
            compounds.append(comp)
            print(f"   ✓ {comp}")
        except Exception as e:
            print(f"   ✗ Erreur lors du chargement de {name}: {e}")
            return
    
    # Package thermodynamique
    thermo = ThermodynamicPackage(compounds)
    
    # Conditions opératoires
    P = 101325  # Pa (1 atm)
    print(f"\n   Pression: {P/1000:.2f} kPa")
    
    # Alimentation
    F = 100.0  # kmol/h
    z_F = np.array([0.333, 0.333, 0.334])  # 33.3% chacun
    
    print(f"   Débit alimentation: {F:.1f} kmol/h")
    print("   Composition alimentation:")
    for i, name in enumerate(compound_names):
        print(f"      • {name:10s}: {z_F[i]*100:.1f}%")
    
    # Afficher les propriétés à une température moyenne
    T_avg = np.mean([comp.Tb for comp in compounds])
    thermo.print_properties(T_avg, P)
    
    # ========================================================================
    # 2. DIMENSIONNEMENT PAR MÉTHODES SIMPLIFIÉES
    # ========================================================================
    print("\n2. DIMENSIONNEMENT PAR MÉTHODES SIMPLIFIÉES")
    print("=" * 80)
    
    # Créer l'objet de dimensionnement
    shortcut = ShortcutDistillation(thermo, F, z_F, P)
    
    # Spécifications de séparation
    recovery_LK_D = 0.95  # 95% du benzène dans distillat
    recovery_HK_B = 0.95  # 95% du toluène dans résidu
    R_factor = 1.3  # R = 1.3 × R_min
    q = 1.0  # Alimentation liquide saturée
    efficiency = 0.70  # Efficacité des plateaux 70%
    
    print(f"\nSpécifications:")
    print(f"   • Récupération benzène (distillat): {recovery_LK_D*100:.0f}%")
    print(f"   • Récupération toluène (résidu):    {recovery_HK_B*100:.0f}%")
    print(f"   • Facteur de reflux:                 {R_factor}")
    print(f"   • Qualité alimentation:              q = {q}")
    print(f"   • Efficacité plateaux:               {efficiency*100:.0f}%")
    
    # Dimensionnement complet
    results = shortcut.complete_shortcut_design(
        recovery_LK_D=recovery_LK_D,
        recovery_HK_B=recovery_HK_B,
        R_factor=R_factor,
        q=q,
        efficiency=efficiency
    )
    
    # ========================================================================
    # 3. RÉSUMÉ DES RÉSULTATS
    # ========================================================================
    print("\n3. RÉSUMÉ DES RÉSULTATS")
    print_design_summary(results, compound_names)
    
    # ========================================================================
    # 4. ESTIMATION DES PROFILS
    # ========================================================================
    print("\n4. ESTIMATION DES PROFILS DE COMPOSITION ET TEMPÉRATURE")
    print("-" * 80)
    
    N_real = results['N_real']
    stages = np.arange(1, N_real + 1)
    
    # Profils de composition (estimation linéaire)
    x_profiles = np.zeros((N_real, len(compounds)))
    y_profiles = np.zeros((N_real, len(compounds)))
    temperatures = np.zeros(N_real)
    
    for j, stage in enumerate(stages):
        # Interpolation linéaire entre distillat et résidu
        if stage <= results['feed_stage']:
            # Section rectification
            ratio = (stage - 1) / results['feed_stage']
            x_stage = results['x_D'] + ratio * (z_F - results['x_D'])
        else:
            # Section épuisement
            ratio = (stage - results['feed_stage']) / (N_real - results['feed_stage'])
            x_stage = z_F + ratio * (results['x_B'] - z_F)
        
        x_stage = x_stage / np.sum(x_stage)  # Normalisation
        x_profiles[j, :] = x_stage
        
        # Température de bulle
        try:
            T_bubble, y_stage = thermo.bubble_temperature(P, x_stage)
            temperatures[j] = T_bubble
            y_profiles[j, :] = y_stage
        except:
            # En cas d'échec, estimation simple
            temperatures[j] = compounds[0].Tb + \
                            (compounds[-1].Tb - compounds[0].Tb) * (j / N_real)
            y_profiles[j, :] = x_stage
    
    print(f"   ✓ Profils estimés pour {N_real} plateaux")
    print(f"   • Température tête:  {temperatures[0]-273.15:.1f}°C")
    print(f"   • Température fond:  {temperatures[-1]-273.15:.1f}°C")
    print(f"   • ΔT colonne:        {(temperatures[-1]-temperatures[0]):.1f} K")
    
    # ========================================================================
    # 5. VISUALISATIONS
    # ========================================================================
    print("\n5. GÉNÉRATION DES VISUALISATIONS")
    print("-" * 80)
    
    visualizer = DistillationVisualizer(compound_names)
    
    # Bilans matières
    print("   Génération: bilans matières...")
    visualizer.plot_material_balance(
        F, results['D'], results['B'],
        z_F, results['x_D'], results['x_B'],
        save_path='btx_bilan_matiere.png'
    )
    
    # Résultats shortcut
    print("   Génération: résultats dimensionnement...")
    visualizer.plot_shortcut_results(
        results,
        save_path='btx_shortcut_results.png'
    )
    
    # Profils de composition (matplotlib)
    print("   Génération: profils de composition (matplotlib)...")
    visualizer.plot_composition_profiles_matplotlib(
        stages, x_profiles, y_profiles,
        results['feed_stage'],
        save_path='btx_composition_profiles.png'
    )
    
    # Profils de composition (plotly interactif)
    print("   Génération: profils de composition (plotly interactif)...")
    try:
        visualizer.plot_composition_profiles_plotly(
            stages, x_profiles, y_profiles,
            results['feed_stage']
        )
    except Exception as e:
        print(f"   ⚠ Visualisation Plotly non disponible: {e}")
    
    # Profil de température
    print("   Génération: profil de température...")
    visualizer.plot_temperature_profile(
        stages, temperatures,
        results['feed_stage'],
        save_path='btx_temperature_profile.png'
    )
    
    # ========================================================================
    # 6. ANALYSE DES RÉSULTATS
    # ========================================================================
    print("\n6. ANALYSE DES RÉSULTATS")
    print("=" * 80)
    
    print("\nDistribution des composés:")
    print(f"{'Composé':<15} {'Alim (kmol/h)':<15} {'Dist (kmol/h)':<15} "
          f"{'Rés (kmol/h)':<15} {'Récup D (%)':<12}")
    print("-" * 80)
    
    for i, name in enumerate(compound_names):
        F_i = F * z_F[i]
        D_i = results['D'] * results['x_D'][i]
        B_i = results['B'] * results['x_B'][i]
        recovery = (D_i / F_i) * 100 if F_i > 0 else 0
        
        print(f"{name:<15} {F_i:<15.2f} {D_i:<15.2f} {B_i:<15.2f} {recovery:<12.1f}")
    
    print("-" * 80)
    print(f"{'TOTAL':<15} {F:<15.2f} {results['D']:<15.2f} "
          f"{results['B']:<15.2f}")
    
    # Vérification des bilans
    print("\nVérification des bilans matières:")
    error = abs(F - results['D'] - results['B'])
    print(f"   Erreur globale: {error:.2e} kmol/h")
    
    for i, name in enumerate(compound_names):
        F_i = F * z_F[i]
        D_i = results['D'] * results['x_D'][i]
        B_i = results['B'] * results['x_B'][i]
        error_i = abs(F_i - D_i - B_i)
        print(f"   Erreur {name:10s}: {error_i:.2e} kmol/h")
    
    # ========================================================================
    # 7. ESTIMATION ÉNERGÉTIQUE
    # ========================================================================
    print("\n7. ESTIMATION ÉNERGÉTIQUE")
    print("=" * 80)
    
    # Température moyenne de tête et fond
    T_top = temperatures[0]
    T_bottom = temperatures[-1]
    
    # Enthalpies
    H_V_top = thermo.mixture_enthalpy_vapor(T_top, results['x_D'])
    H_L_top = thermo.mixture_enthalpy_liquid(T_top, results['x_D'])
    H_V_bottom = thermo.mixture_enthalpy_vapor(T_bottom, results['x_B'])
    H_L_bottom = thermo.mixture_enthalpy_liquid(T_bottom, results['x_B'])
    
    # Chaleur de condensation
    V = results['V']  # kmol/h
    Q_condenser = V * (H_V_top - H_L_top) / 1000  # kW (approximation)
    
    # Chaleur de rebouillage
    V_bottom = V  # Approximation CMO
    Q_reboiler = V_bottom * (H_V_bottom - H_L_bottom) / 1000  # kW
    
    print(f"\nBesoins énergétiques (estimation):")
    print(f"   • Condenseur:  {abs(Q_condenser):.1f} kW (refroidissement)")
    print(f"   • Rebouilleur: {Q_reboiler:.1f} kW (chauffage)")
    print(f"   • Rapport Q_R/Q_C: {Q_reboiler/abs(Q_condenser):.2f}")
    
    # Consommation vapeur (vapeur à 3 bar ≈ 2100 kJ/kg)
    latent_heat_steam = 2100  # kJ/kg
    steam_consumption = Q_reboiler / latent_heat_steam * 3600  # kg/h
    
    print(f"\nConsommation de vapeur (3 bar):")
    print(f"   • {steam_consumption:.1f} kg/h")
    print(f"   • Ratio vapeur/alimentation: {steam_consumption/(F*80):.2f} kg_vapeur/kg_produit")
    
    # ========================================================================
    # 8. CONCLUSION
    # ========================================================================
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + "DIMENSIONNEMENT TERMINÉ AVEC SUCCÈS".center(78) + "║")
    print("╚" + "═" * 78 + "╝")
    
    print("\nFichiers générés:")
    print("   ✓ btx_bilan_matiere.png")
    print("   ✓ btx_shortcut_results.png")
    print("   ✓ btx_composition_profiles.png")
    print("   ✓ btx_temperature_profile.png")
    print("   ✓ composition_profiles_interactive.html (si Plotly disponible)")
    
    print("\nPour une simulation plus précise, utiliser:")
    print("   → Méthode MESH rigoureuse (mesh_solver.py)")
    print("   → Validation avec Aspen Plus")
    print("   → Optimisation des paramètres")
    
    return results, thermo, visualizer

def etude_parametrique_reflux():
    """
    Étude de l'effet du reflux sur le nombre de plateaux
    """
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + "ÉTUDE PARAMÉTRIQUE: EFFET DU REFLUX".center(78) + "║")
    print("╚" + "═" * 78 + "╝\n")
    
    # Système BTX
    compound_names = ['benzene', 'toluene', 'o-xylene']
    compounds = [Compound(name) for name in compound_names]
    thermo = ThermodynamicPackage(compounds)
    
    F = 100.0
    z_F = np.array([0.333, 0.333, 0.334])
    P = 101325
    
    shortcut = ShortcutDistillation(thermo, F, z_F, P)
    D, B, x_D, x_B = shortcut.material_balance()
    N_min, alpha_avg = shortcut.fenske_equation()
    R_min, theta = shortcut.underwood_method()
    
    # Varier le reflux
    R_factors = np.linspace(1.1, 3.0, 20)
    N_values = []
    
    for factor in R_factors:
        R = factor * R_min
        N = shortcut.gilliland_correlation(R)
        N_values.append(N / 0.70)  # Avec efficacité 70%
    
    # Visualisation
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.plot(R_factors, N_values, 'b-', linewidth=3, label='Courbe N vs R/R_min')
    ax.axhline(y=N_min/0.70, color='r', linestyle='--', linewidth=2,
              label=f'N_min = {N_min/0.70:.1f}')
    ax.axvline(x=1.3, color='g', linestyle='--', linewidth=2,
              label='R = 1.3×R_min (typique)')
    
    # Point optimal (approximation)
    idx_opt = np.argmin(0.4*np.array(N_values)/N_values[0] + 0.6*R_factors)
    ax.plot(R_factors[idx_opt], N_values[idx_opt], 'ro', markersize=12,
           label=f'Optimum économique (R/R_min ≈ {R_factors[idx_opt]:.2f})')
    
    ax.set_xlabel('R / R_min', fontsize=12, fontweight='bold')
    ax.set_ylabel('Nombre de plateaux réels', fontsize=12, fontweight='bold')
    ax.set_title('Effet du rapport de reflux sur le nombre de plateaux\n(Système BTX)',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([1.0, 3.0])
    
    plt.tight_layout()
    plt.savefig('btx_etude_reflux.png', dpi=300, bbox_inches='tight')
    print("✓ Graphique sauvegardé: btx_etude_reflux.png")
    plt.show()
    
    print(f"\nRésultats de l'étude:")
    print(f"   • N_min (E=70%):     {N_min/0.70:.1f} plateaux")
    print(f"   • R_min:             {R_min:.3f}")
    print(f"   • R optimal (est.):  {R_factors[idx_opt]*R_min:.3f}")
    print(f"   • N @ R_opt:         {N_values[idx_opt]:.1f} plateaux")

if __name__ == "__main__":
    """
    Point d'entrée principal
    """
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "MODÉLISATION ET SIMULATION DE DISTILLATION MULTICOMPOSANTS".center(78) + "█")
    print("█" + "Cours: Modélisation et Simulation des Procédés - PIC".center(78) + "█")
    print("█" + "Prof. BAKHER Zine Elabidine - UM6P".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80 + "\n")
    
    try:
        # Exemple principal
        results, thermo, visualizer = exemple_btx_complet()
        
        # Étude paramétrique
        print("\n" + "─" * 80)
        input("\nAppuyez sur Entrée pour lancer l'étude paramétrique du reflux...")
        etude_parametrique_reflux()
        
        print("\n" + "█" * 80)
        print("█" + "SIMULATION COMPLÉTÉE AVEC SUCCÈS".center(78) + "█")
        print("█" * 80 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Simulation interrompue par l'utilisateur")
    except Exception as e:
        print(f"\n\n✗ Erreur lors de la simulation: {e}")
        import traceback
        traceback.print_exc()