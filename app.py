"""
Application Flask pour la Distillation Multicomposants
======================================================
Basée sur le cours: Modélisation et Simulation des Procédés - PIC
Prof. BAKHER Zine Elabidine - Université UH1

Référence: Support PDF - Distillation de Mélanges Multicomposants
"""

from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from distillation_multicomposants import (
    Compound, ThermodynamicPackage, ShortcutDistillation
)
from visualization import DistillationVisualizer
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'distillation-pic-uh1-2024'

# ComposÃ©s disponibles (Tableau 6 - PropriÃ©tÃ©s BTX du PDF)
AVAILABLE_COMPOUNDS = {
    'benzene': 'Benzène (Câ‚†Hâ‚†)',
    'toluene': 'Toluène (Câ‚‡Hâ‚ˆ)',
    'o-xylene': 'o-Xylène (Câ‚ˆHâ‚â‚€)',
    'ethylbenzene': 'éthylbenzène',
    'propane': 'Propane',
    'butane': 'Butane',
    'pentane': 'Pentane',
    'hexane': 'Hexane',
    'heptane': 'Heptane',
    'octane': 'Octane',
    'methanol': 'Méthanol',
    'ethanol': 'ethanol',
    'propanol': 'Propanol',
    'water': 'Eau',
    'acetone': 'Acétone',
}

def fig_to_base64(fig):
    """Convertit une figure matplotlib en base64"""
    img = io.BytesIO()
    fig.savefig(img, format='png', dpi=150, bbox_inches='tight')
    img.seek(0)
    plt.close(fig)
    return base64.b64encode(img.getvalue()).decode()

@app.route('/')
def index():
    """Page d'accueil"""
    return render_template('app.html', compounds=AVAILABLE_COMPOUNDS)

@app.route('/api/simulate', methods=['POST'])
def simulate():
    """
    Endpoint pour lancer une simulation
    Implemente les méthodes du PDF:
    - Section 3: Méthodes simplifiées (Fenske, Underwood, Gilliland, Kirkbride)
    - Section 4: Méthode MESH (optionnel)
    """
    try:
        data = request.json
        
        # RÃ©cupÃ©ration des paramÃ¨tres
        compound_names = data.get('compounds', ['benzene', 'toluene', 'o-xylene'])
        compositions = data.get('compositions', [0.333, 0.333, 0.334])
        F = float(data.get('flow_rate', 100))
        P = float(data.get('pressure', 101325))
        recovery_LK = float(data.get('recovery_lk', 0.95))
        recovery_HK = float(data.get('recovery_hk', 0.95))
        R_factor = float(data.get('reflux_factor', 1.3))
        efficiency = float(data.get('efficiency', 0.70))
        q = float(data.get('feed_quality', 1.0))
        
        # Normalisation des compositions (3 chiffres aprÃ¨s virgule)
        z_F = np.array(compositions)
        z_F = z_F / np.sum(z_F)
        z_F = np.round(z_F, 3)
        
        # CrÃ©ation des composÃ©s
        compounds = []
        for name in compound_names:
            try:
                comp = Compound(name)
                compounds.append(comp)
            except Exception as e:
                return jsonify({'error': f'Erreur chargement {name}: {str(e)}'}), 400
        
        # Package thermodynamique (Section 2.1 du PDF)
        thermo = ThermodynamicPackage(compounds)
        
        # Dimensionnement shortcut (Section 3 du PDF)
        shortcut = ShortcutDistillation(thermo, F, z_F, P)
        
        # ExÃ©cution des mÃ©thodes simplifiÃ©es
        results = shortcut.complete_shortcut_design(
            recovery_LK_D=recovery_LK,
            recovery_HK_B=recovery_HK,
            R_factor=R_factor,
            q=q,
            efficiency=efficiency
        )
        
        # Calcul des volatilitÃ©s relatives (Section 2.2 du PDF)
        T_avg = np.mean([comp.Tb for comp in compounds])
        alpha = thermo.relative_volatilities(T_avg, P)
        
        # GÃ©nÃ©ration des profils de composition
        N_real = results['N_real']
        stages = np.arange(1, N_real + 1)
        x_profiles = np.zeros((N_real, len(compounds)))
        y_profiles = np.zeros((N_real, len(compounds)))
        temperatures = np.zeros(N_real)
        
        # Calcul des profils (interpolation linÃ©aire)
        for j, stage in enumerate(stages):
            if stage <= results['feed_stage']:
                ratio = (stage - 1) / max(results['feed_stage'], 1)
                x_stage = results['x_D'] + ratio * (z_F - results['x_D'])
            else:
                ratio = (stage - results['feed_stage']) / max((N_real - results['feed_stage']), 1)
                x_stage = z_F + ratio * (results['x_B'] - z_F)
            
            x_stage = x_stage / np.sum(x_stage)
            x_stage = np.round(x_stage, 3)
            x_profiles[j, :] = x_stage
            
            # Calcul tempÃ©rature de bulle (Section 2.3.1 du PDF)
            try:
                T_bubble, y_stage = thermo.bubble_temperature(P, x_stage)
                temperatures[j] = T_bubble
                y_stage = np.round(y_stage, 3)
                y_profiles[j, :] = y_stage
            except:
                temperatures[j] = compounds[0].Tb + (compounds[-1].Tb - compounds[0].Tb) * (j / N_real)
                y_profiles[j, :] = x_stage
        
        # ============================================
        # VISUALISATIONS (Section 6 du PDF)
        # ============================================
        
        visualizer = DistillationVisualizer(compound_names)
        
        # 1. Bilans matiÃ¨res (Section 4.2.1 du PDF - Ã‰quation M)
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig1.suptitle('Bilans Matières - équations MESH (M)', fontweight='bold', fontsize=13)
        
        streams = ['Alimentation\n(F)', 'Distillat\n(D)', 'Résidu\n(B)']
        flows = [F, results['D'], results['B']]
        colors_streams = ['#3182CE', '#38A169', '#E53E3E']
        
        bars = ax1.bar(streams, flows, color=colors_streams, alpha=0.85, 
                      edgecolor='black', linewidth=1.5, width=0.6)
        
        for bar, flow in zip(bars, flows):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{flow:.3f} kmol/h',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax1.set_ylabel('Débit (kmol/h)', fontweight='bold', fontsize=11)
        ax1.set_title('Débits des flux', fontweight='bold', fontsize=11)
        ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax1.set_ylim([0, max(flows) * 1.15])
        
        # Compositions
        x = np.arange(len(compounds))
        width = 0.25
        
        # Arrondir Ã  3 dÃ©cimales
        z_F_plot = np.round(z_F, 3)
        x_D_plot = np.round(results['x_D'], 3)
        x_B_plot = np.round(results['x_B'], 3)
        
        bars1 = ax2.bar(x - width, z_F_plot, width, label='Alimentation (z_F)',
                       color='#3182CE', alpha=0.85, edgecolor='black', linewidth=1)
        bars2 = ax2.bar(x, x_D_plot, width, label='Distillat (x_D)',
                       color='#38A169', alpha=0.85, edgecolor='black', linewidth=1)
        bars3 = ax2.bar(x + width, x_B_plot, width, label='Résidu (x_B)',
                       color='#E53E3E', alpha=0.85, edgecolor='black', linewidth=1)
        
        # Afficher valeurs sur les barres
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0.001:
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.3f}',
                            ha='center', va='bottom', fontsize=8)
        
        ax2.set_xlabel('Composans ', fontweight='bold', fontsize=11)
        ax2.set_ylabel('Fraction molaire', fontweight='bold', fontsize=11)
        ax2.set_title('Compositions (3 décimales)', fontweight='bold', fontsize=11)
        ax2.set_xticks(x)
        ax2.set_xticklabels([AVAILABLE_COMPOUNDS.get(n, n).split('(')[0].strip() 
                            for n in compound_names], rotation=15, ha='right')
        ax2.legend(fontsize=9, loc='upper right')
        ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax2.set_ylim([0, 1.0])
        
        plt.tight_layout()
        plot1 = fig_to_base64(fig1)
        
        # 2. Profils de composition (Section 7.3 du PDF)
        fig2, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.Set3(np.linspace(0, 1, len(compounds)))
        
        for i in range(len(compounds)):
            ax.plot(x_profiles[:, i], stages, 'o-', linewidth=2.5,
                   label=AVAILABLE_COMPOUNDS.get(compound_names[i], compound_names[i]).split('(')[0].strip(),
                   color=colors[i], markersize=5)
        
        # Ligne plateau alimentation (Ã‰quation de Kirkbride - Section 3.4)
        ax.axhline(y=results['feed_stage'], color='blue', linestyle='--', linewidth=2,
                  label=f'Plateau alimentation (Kirkbride): {results["feed_stage"]}')
        
        ax.set_xlabel('Fraction molaire liquide (x)', fontweight='bold', fontsize=11)
        ax.set_ylabel('Numéro de plateau', fontweight='bold', fontsize=11)
        ax.set_title('Profils de Composition - Phase Liquide\n(Méthode simplifiée)', 
                    fontweight='bold', fontsize=12)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.invert_yaxis()
        ax.set_xlim([0, 1])
        
        # Annotations zones
        ax.text(0.95, results['N_R']/2, f'Rectification\n({results["N_R"]} plateaux)', 
                ha='right', va='center', fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        ax.text(0.95, results['feed_stage'] + results['N_S']/2, 
                f'épuisement\n({results["N_S"]} plateaux)', 
                ha='right', va='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        plt.tight_layout()
        plot2 = fig_to_base64(fig2)
        
        # 3. Profil de tempÃ©rature (Section 7.4 du PDF)
        fig3, ax = plt.subplots(figsize=(8, 8))
        
        ax.plot(temperatures - 273.15, stages, 'o-', linewidth=3,
               markersize=7, color='#E53E3E', label='Température')
        ax.axhline(y=results['feed_stage'], color='blue', linestyle='--', linewidth=2,
                  label=f'Plateau alimentation: {results["feed_stage"]}')
        
        ax.set_xlabel('Température (Â°C)', fontweight='bold', fontsize=11)
        ax.set_ylabel('Numéro de plateau', fontweight='bold', fontsize=11)
        ax.set_title('Profil de Température dans la Colonne', fontweight='bold', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.invert_yaxis()
        
        # Annotations tempÃ©ratures clÃ©s
        T_top = temperatures[0] - 273.15
        T_bottom = temperatures[-1] - 273.15
        ax.text(T_top, 1, f' Tete: {T_top:.1f}Â°C', ha='left', va='center',
               fontweight='bold', color='darkred', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.text(T_bottom, len(stages), f' Fond: {T_bottom:.1f}°C', ha='left', va='center',
               fontweight='bold', color='darkred', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plot3 = fig_to_base64(fig3)
        
        # PrÃ©paration de la rÃ©ponse
        response = {
            'success': True,
            'results': {
                # MÃ©thode de Fenske (Section 3.1)
                'N_min': round(float(results['N_min']), 2),
                'alpha_avg': round(float(results['alpha_avg']), 3),
                
                # MÃ©thode d'Underwood (Section 3.2)
                'R_min': round(float(results['R_min']), 3),
                'theta': round(float(results['theta']), 3),
                
                # CorrÃ©lation de Gilliland (Section 3.3)
                'N_theoretical': round(float(results['N_theoretical']), 2),
                'N_real': int(results['N_real']),
                'R': round(float(results['R']), 3),
                
                # Ã‰quation de Kirkbride (Section 3.4)
                'feed_stage': int(results['feed_stage']),
                'N_R': int(results['N_R']),
                'N_S': int(results['N_S']),
                
                # Bilans matiÃ¨res
                'D': round(float(results['D']), 3),
                'B': round(float(results['B']), 3),
                'x_D': [round(float(x), 3) for x in results['x_D']],
                'x_B': [round(float(x), 3) for x in results['x_B']],
                'z_F': [round(float(x), 3) for x in z_F],
                
                # ParamÃ¨tres
                'efficiency': float(results['efficiency']),
                'L': round(float(results['L']), 3),
                'V': round(float(results['V']), 3),
                
                # TempÃ©ratures
                'T_top': round(float(temperatures[0] - 273.15), 1),
                'T_bottom': round(float(temperatures[-1] - 273.15), 1),
                
                # VolatilitÃ©s relatives (Section 2.2)
                'volatilities': [round(float(a), 3) for a in alpha],
                
                # RÃ©cupÃ©rations
                'recovery_LK_D': round(recovery_LK * 100, 1),
                'recovery_HK_B': round(recovery_HK * 100, 1),
            },
            'plots': {
                'material_balance': plot1,
                'composition_profile': plot2,
                'temperature_profile': plot3
            },
            'compound_names': [AVAILABLE_COMPOUNDS.get(n, n) for n in compound_names]
        }
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/compound_info/<compound>')
def compound_info(compound):
    """Récupere les informations d'un composant  (Tableau 6 du PDF)"""
    try:
        comp = Compound(compound)
        info = {
            'name': compound,
            'display_name': AVAILABLE_COMPOUNDS.get(compound, compound),
            'Tb': round(float(comp.Tb - 273.15), 2),  # Â°C
            'Tc': round(float(comp.Tc - 273.15), 2),  # Â°C
            'Pc': round(float(comp.Pc / 1e5), 2),     # bar
            'MW': round(float(comp.MW), 2),           # g/mol
            'omega': round(float(comp.omega), 3) if comp.omega else None
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/available_compounds')
def available_compounds():
    """Liste des composans disponibles"""
    return jsonify(AVAILABLE_COMPOUNDS)

@app.errorhandler(404)
def not_found(e):
    return render_template('app.html', compounds=AVAILABLE_COMPOUNDS), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Erreur serveur interne'}), 500

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    print("\n" + "="*80)
    print("  APPLICATION FLASK - DISTILLATION MULTICOMPOSANTS".center(80))
    print("  Oumssaad el ghazi ".center(80))
    print("  Basé sur le support PDF - Année Universitaire 2024-2025".center(80))
    print("="*80 + "\n")
    print(" *** Démarrage du serveur...")
    print(" ***  Interface disponible sur: http://127.0.0.1:5000")
    print(" *** Méthodes implémentées:")
    print("   - Section 3.1: Méthode de Fenske (N_min)")
    print("   - Section 3.2: Méthode d'Underwood (R_min)")
    print("   - Section 3.3: Corrélation de Gilliland (N)")
    print("   - Section 3.4: équation de Kirkbride (Feed stage)")
    print("   - Section 4: équations MESH (bilans)")
    print("\n" + "="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)