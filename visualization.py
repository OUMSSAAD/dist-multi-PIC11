import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------------------
# PRINT SUMMARY (robuste aux clés manquantes / alias)
# --------------------------------------------------------------------------------------

def _safe_get(results, *keys, default=None):
    """
    Retourne la première valeur trouvée parmi les clés fournies,
    ou default si aucune n'existe.
    """
    for k in keys:
        if k in results:
            return results[k]
    return default

def print_design_summary(results, compound_names=None):
    """
    Affiche un rapport lisible à partir du dictionnaire results.
    Utilise des accès robustes (pas d'KeyError si une clé manque).
    """
    print("\n" + "=" * 80)
    print("DIMENSIONNEMENT PAR MÉTHODES SIMPLIFIÉES")
    print("=" * 80)

    # 1. Bilans matières
    D = _safe_get(results, 'D')
    B = _safe_get(results, 'B')
    print("\n1. BILANS MATIÈRES GLOBAUX")
    if D is not None:
        print(f"   Débit distillat: D = {D:.2f} kmol/h")
    else:
        print("   Débit distillat: D = N/A")
    if B is not None:
        print(f"   Débit résidu:    B = {B:.2f} kmol/h")
    else:
        print("   Débit résidu:    B = N/A")

    # 2. Fenske
    print("\n2. ÉQUATION DE FENSKE (reflux total)")
    N_min = _safe_get(results, 'N_min')
    alpha_avg = _safe_get(results, 'alpha_avg', 'alpha')
    if N_min is not None:
        print(f"   N_min = {N_min:.2f} plateaux")
    else:
        print("   N_min = N/A")
    if alpha_avg is not None:
        print(f"   α_avg(LK/HK) = {alpha_avg:.3f}")
    else:
        print("   α_avg(LK/HK) = N/A")

    # 3. Underwood
    print("\n3. MÉTHODE D'UNDERWOOD (reflux minimum)")
    R_min = _safe_get(results, 'R_min')
    theta = _safe_get(results, 'theta')
    if R_min is not None:
        print(f"   R_min = {R_min:.3f}")
    else:
        print("   R_min = N/A")
    if theta is not None:
        print(f"   θ = {theta:.3f}")
    else:
        print("   θ = N/A")

    # 4. Gilliland
    print("\n4. CORRÉLATION DE GILLILAND")
    R = _safe_get(results, 'R')
    # chercher 'R_factor' s'il existe sinon tenter de le reconstituer
    R_factor = _safe_get(results, 'R_factor')
    if R_factor is None and (R is not None) and (R_min is not None) and R_min != 0:
        try:
            R_factor = R / R_min
        except Exception:
            R_factor = None

    N_theoretical = _safe_get(results, 'N_theoretical')
    efficiency = _safe_get(results, 'efficiency', default=None)
    N_real = _safe_get(results, 'N_real')

    if R is not None:
        if R_factor is not None:
            print(f"   R opératoire = {R:.3f} ({R_factor:.3f}× R_min)")
        else:
            print(f"   R opératoire = {R:.3f}")
    else:
        print("   R opératoire = N/A")

    if N_theoretical is not None:
        print(f"   N théorique = {N_theoretical:.2f} plateaux")
    else:
        print("   N théorique = N/A")

    if efficiency is not None:
        print(f"   Efficacité = {efficiency * 100:.1f}%")
    else:
        print("   Efficacité = N/A")

    if N_real is not None:
        print(f"   N réel = {N_real} plateaux")
    else:
        print("   N réel = N/A")

    # 5. Kirkbride / position alimentation
    print("\n5. ÉQUATION DE KIRKBRIDE (position alimentation)")
    # accepter plusieurs variantes de noms
    N_R = _safe_get(results, 'N_R', 'NR', default=None)
    N_S = _safe_get(results, 'N_S', 'NS', default=None)
    feed_stage = _safe_get(results, 'feed_stage')
    if N_R is not None:
        print(f"   Plateaux rectification: {N_R}")
    else:
        print("   Plateaux rectification: N/A")
    if N_S is not None:
        print(f"   Plateaux épuisement:    {N_S}")
    else:
        print("   Plateaux épuisement:    N/A")
    if feed_stage is not None:
        print(f"   Plateau d'alimentation: {feed_stage}")
    else:
        print("   Plateau d'alimentation: N/A")

    # 6. Débits internes (robuste)
    print("\n6. DÉBITS INTERNES")
    # noms alternatifs: L/V vs L_prime/V_prime vs L_stripping/V_stripping
    L = _safe_get(results, 'L')
    V = _safe_get(results, 'V')
    L_prime = _safe_get(results, 'L_prime', 'L_stripping')
    V_prime = _safe_get(results, 'V_prime', 'V_stripping')

    if L is not None:
        print(f"   Liquide rectification L = {L:.2f} kmol/h")
    else:
        print("   Liquide rectification L = N/A")

    if V is not None:
        print(f"   Vapeur rectification  V = {V:.2f} kmol/h")
    else:
        print("   Vapeur rectification  V = N/A")

    if L_prime is not None:
        print(f"   Liquide épuisement L' = {L_prime:.2f} kmol/h")
    else:
        print("   Liquide épuisement L' = N/A")

    if V_prime is not None:
        print(f"   Vapeur épuisement  V' = {V_prime:.2f} kmol/h")
    else:
        print("   Vapeur épuisement  V' = N/A")

    # Optionnel: distribution composés (si x_D/x_B disponibles)
    x_D = _safe_get(results, 'x_D')
    x_B = _safe_get(results, 'x_B')
    if x_D is not None and x_B is not None and compound_names is not None:
        print("\nDistribution approximative (fractions molaires):")
        for i, name in enumerate(compound_names):
            xd = x_D[i] if i < len(x_D) else 0.0
            xb = x_B[i] if i < len(x_B) else 0.0
            print(f"   {name:<12s} x_D = {xd:.4f}   x_B = {xb:.4f}")

    print("\n" + "=" * 80)


# --------------------------------------------------------------------------------------
# VISUALISATIONS
# --------------------------------------------------------------------------------------

class DistillationVisualizer:

    def __init__(self, names):
        self.names = names

    def plot_material_balance(self, F, D, B, zF, xD, xB, save_path="material_balance.png"):
        labels = self.names if self.names is not None else ["C1", "C2", "C3"]
        fig, ax = plt.subplots()

        ax.bar(labels, zF * F, label="Alimentation")
        ax.bar(labels, xD * D, label="Distillat")
        ax.bar(labels, xB * B, label="Résidu")

        ax.legend()
        ax.set_title("Bilan Matière BTX")

        plt.savefig(save_path)
        plt.close()

    def plot_shortcut_results(self, results, save_path="shortcut_results.png"):
        fig, ax = plt.subplots()
        ax.axis("off")

        # construction du texte avec sécurité sur les clés
        lines = []
        lines.append(f"N_min = {_safe_get(results,'N_min','N_min')}")
        lines.append(f"R_min = {_safe_get(results,'R_min','N/A')}")
        lines.append(f"N théorique = {_safe_get(results,'N_theoretical','N/A')}")
        lines.append(f"N réel = {_safe_get(results,'N_real','N/A')}")
        lines.append(f"Plateau alimentation = {_safe_get(results,'feed_stage','N/A')}")

        text = "\n".join(str(l) for l in lines)
        ax.text(0.1, 0.5, text, fontsize=12, family='monospace')
        plt.savefig(save_path)
        plt.close()

    def plot_temperature_profile(self, stages, T, feed_stage, save_path="temperature_profile.png"):
        plt.plot(stages, T, "-o")
        if feed_stage is not None:
            plt.axvline(feed_stage, color="red", linestyle="--")
        plt.xlabel("Plateau")
        plt.ylabel("Température (K)")
        plt.title("Profil de Température")
        plt.savefig(save_path)
        plt.close()


# --------------------------------------------------------------------------------------
# ➕ NOUVELLE FONCTION : capture du rapport complet
# --------------------------------------------------------------------------------------

def get_design_summary_text(results, compound_names=None):
    """
    Capture la sortie de print_design_summary() et renvoie le texte.
    Utilise la fonction print_design_summary robuste ci-dessus.
    """
    from io import StringIO
    import sys

    buffer = StringIO()
    temp_stdout = sys.stdout
    sys.stdout = buffer

    try:
        print_design_summary(results, compound_names)
    except Exception as e:
        # En cas d'erreur imprévue, imprimer un message lisible
        print(f"Erreur lors de la génération du rapport: {e}")
    finally:
        sys.stdout = temp_stdout

    return buffer.getvalue()
