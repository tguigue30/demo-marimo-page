# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.17.0",
#     "matplotlib>=3.10.7",
#     "numpy>=2.3.5",
#     "pyzmq>=27.1.0",
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from math import erf, sqrt
    return erf, mo, np, plt, sqrt


@app.cell
def _(mo):
    titre = mo.md(
        r"""
    # Inférence statistique interactive

    Cette mini-application illustre les idées centrales de l’inférence statistique :

    - **On observe un échantillon** tiré d’une population (normale ou Bernoulli)  
    - On calcule des **estimateurs** (par ex. la moyenne empirique)  
    - On construit un **intervalle de confiance** pour le paramètre inconnu  
    - On réalise un **test d’hypothèse** (*H₀: paramètre = valeur de référence*)  

    Utilise les widgets ci-dessous pour faire varier :

    - la loi de la population (normale / Bernoulli),
    - la taille d’échantillon,
    - les paramètres vrais,
    - la valeur testée dans l’hypothèse nulle,
    - le niveau de significativité du test.
    """
    )
    titre
    return


@app.cell
def _(mo):
    # Choix de la loi
    dist_radio = mo.ui.radio(
        options=["Normale", "Bernoulli"],
        value="Normale",
        label="Loi de la population",
    )

    # Taille d'échantillon
    n_slider = mo.ui.slider(
        start=5,
        stop=500,
        step=5,
        value=50,
        label="Taille de l'échantillon n",
    )

    # Paramètres pour la loi normale
    mu_true = mo.ui.number(
        value=0.0,
        step=0.1,
        label="μ vrai (moyenne vraie, loi normale)",
    )
    sigma_true = mo.ui.number(
        value=1.0,
        step=0.1,
        label="σ vrai (écart-type, loi normale)",
    )

    # Paramètre pour la Bernoulli
    p_true = mo.ui.slider(
        start=0.05,
        stop=0.95,
        step=0.05,
        value=0.5,
        label="p vrai (probabilité de succès, loi de Bernoulli)",
    )

    # Paramètre de l'hypothèse nulle (moyenne ou proportion)
    mu0 = mo.ui.number(
        value=0.0,
        step=0.1,
        label="Valeur testée : μ₀ ou p₀ (H₀ : paramètre = μ₀/p₀)",
    )

    # Niveau de significativité
    alpha_slider = mo.ui.slider(
        start=0.01,
        stop=0.20,
        step=0.01,
        value=0.05,
        label="Niveau de significativité α",
    )
    return alpha_slider, dist_radio, mu0, mu_true, n_slider, p_true, sigma_true


@app.cell
def _(
    alpha_slider,
    dist_radio,
    mo,
    mu0,
    mu_true,
    n_slider,
    p_true,
    sigma_true,
):
    controles = mo.hstack(
        [
            mo.vstack(
                [
                    mo.md("### 1. Choix de la loi et de n"),
                    dist_radio,
                    n_slider,
                ]
            ),
            mo.vstack(
                [
                    mo.md("### 2. Paramètres vrais"),
                    mu_true,
                    sigma_true,
                    p_true,
                ]
            ),
            mo.vstack(
                [
                    mo.md("### 3. Test d'hypothèse"),
                    mu0,
                    alpha_slider,
                ]
            ),
        ]
    )
    controles
    return


@app.cell
def _(dist_radio, mu_true, n_slider, np, p_true, sigma_true):
    n = n_slider.value

    if dist_radio.value == "Normale":
        data = np.random.normal(loc=mu_true.value, scale=sigma_true.value, size=n)
        true_param = mu_true.value
        param_name = "μ"
        is_bernoulli = False
    else:
        data = np.random.binomial(n=1, p=p_true.value, size=n)
        true_param = p_true.value
        param_name = "p"
        is_bernoulli = True

    sample_mean = float(data.mean())
    sample_std = float(data.std(ddof=1))  # écart-type empirique
    return (
        data,
        is_bernoulli,
        n,
        param_name,
        sample_mean,
        sample_std,
        true_param,
    )


@app.cell
def _(is_bernoulli, mo, n, param_name, sample_mean, sample_std, true_param):
    nature = "proportion" if is_bernoulli else "moyenne"
    resume = mo.md(
        fr"""
    ### Résumé de l'échantillon

    - Taille de l'échantillon : **n = {n}**  
    - {nature.capitalize()} empirique : **$\hat{{{param_name}}} = {sample_mean:.3f}$**  
    - Écart-type empirique : **$s = {sample_std:.3f}$**  
    - Vrai paramètre : **${param_name}^* = {true_param:.3f}$**  

    On souhaite maintenant **inférer** le paramètre de la population à partir de cet échantillon :
    - Construire un **intervalle de confiance**  
    - Réaliser un **test d'hypothèse** (H₀ : {param_name} = valeur donnée)
    """
    )
    resume
    return


@app.cell
def _(n, sample_mean, sample_std):
    # On fixe un niveau de confiance à 95% pour l'IC (z ≈ 1.96)
    z_95 = 1.96
    se = sample_std / (n ** 0.5)
    ci_low = sample_mean - z_95 * se
    ci_high = sample_mean + z_95 * se
    return ci_high, ci_low, se, z_95


@app.cell
def _(ci_high, ci_low, mo, n, param_name, se, z_95):
    ic_md = mo.md(
        fr"""
    ### Intervalle de confiance à 95 % pour {param_name}

    On utilise ici l'approximation normale (CLT) :

    $$
    \hat{{{param_name}}} \sim \mathcal{{N}}\left({param_name}, \ \frac{{\sigma^2}}{n}\right)
    \approx \mathcal{{N}}\left({param_name}, \ \frac{{s^2}}{n}\right)
    $$

    Un **intervalle de confiance à 95 %** pour {param_name} est donné par :

    $$
    \hat{{{param_name}}} \pm {z_95:.2f} \times \text{{SE}}, 
    \quad \text{{avec SE}} = \frac{{s}}{{\sqrt{{n}}}}
    $$

    Numériquement :

    - Erreur standard : **SE = {se:.3f}**  
    - Intervalle de confiance : **[{ci_low:.3f} ; {ci_high:.3f}]**

    Interprétation (fréquenciste) :  
    *Si l'on répétait indéfiniment l'expérience, environ 95 % des intervalles construits de cette façon contiendraient la vraie valeur du paramètre.*
    """
    )
    ic_md
    return


@app.cell
def _(ci_high, ci_low, data, plt, sample_mean, true_param):
    fig, ax = plt.subplots()

    # Histogramme de l'échantillon
    ax.hist(data, bins=15, density=True)
    ax.set_xlabel("Valeurs observées")
    ax.set_ylabel("Densité (approx.)")
    ax.set_title("Échantillon + intervalle de confiance à 95 %")

    # Moyenne empirique
    ax.axvline(sample_mean, linestyle="--", linewidth=2, label="Moyenne empirique")

    # Vrai paramètre
    ax.axvline(true_param, linestyle="-.", linewidth=2, label="Vrai paramètre")

    # Bornes de l'IC
    ax.axvline(ci_low, linestyle=":", linewidth=2, label="IC 95% - borne inf.")
    ax.axvline(ci_high, linestyle=":", linewidth=2, label="IC 95% - borne sup.")

    ax.legend()

    fig
    return


@app.cell
def _(alpha_slider, erf, mu0, sample_mean, se, sqrt):
    alpha = alpha_slider.value
    # Statistique de test (z-test approx.)
    z_stat = (sample_mean - mu0.value) / se

    # Fonction de répartition de N(0,1)
    def Phi(z):
        return 0.5 * (1 + erf(z / sqrt(2.0)))

    p_value = 2 * (1 - Phi(abs(z_stat)))  # test bilatéral

    # Décision
    reject = p_value < alpha

    conclusion = "On rejette H₀" if reject else "On ne rejette pas H₀"
    return alpha, conclusion, p_value, z_stat


@app.cell
def _(alpha, conclusion, mo, mu0, p_value, param_name, z_stat):
    test_md = mo.md(
        fr"""
    ### Test d'hypothèse sur {param_name}

    On teste l'hypothèse nulle :

    $$
    H_0 : {param_name} = {mu0.value:.3f}
    \quad
    H_1 : {param_name} \neq {mu0.value:.3f}
    $$

    On utilise la statistique de test (approximation normale) :

    $$
    Z = \frac{{\hat{{{param_name}}} - {mu0.value:.3f}}}{{\text{{SE}}}}
    $$

    - Statistique observée : **z = {z_stat:.3f}**  
    - p-value (bilatérale) : **p-value = {p_value:.4f}**  
    - Niveau $\alpha$ choisi : **{alpha:.2f}**

    **Décision :** {conclusion} au niveau $\alpha = {alpha:.2f}$.

    Interprétation :
    - Si l'on **rejette H₀**, les données sont jugées peu compatibles avec la valeur {mu0.value:.3f} pour {param_name}.  
    - Si l'on **ne rejette pas H₀**, les données ne fournissent pas assez d'évidence pour écarter cette valeur (ce qui ne prouve pas que H₀ est vraie).
    """
    )
    test_md
    return


@app.cell
def _(mo):
    theorie = mo.md(
        r"""
    ---

    ## Rappel théorique (très synthétique)

    1. **Statistiques descriptives**  
       On commence par résumer les données : moyenne, variance, histogramme, etc.

    2. **Estimateur**  
       Un estimateur $\hat{\theta}$ d’un paramètre $\theta$ est une fonction de l’échantillon.
       - Exemple : $\hat{\mu} = \frac{1}{n}\sum X_i$ pour la moyenne d'une loi normale.
       - Exemple : $\hat{p} = \frac{1}{n}\sum X_i$ pour la proportion de succès d’une loi de Bernoulli.

    3. **Distribution d'échantillonnage**  
       La loi de $\hat{\theta}$ (quand on répète l’expérience) est fondamentale :
       - Pour de nombreux estimateurs, le **théorème central limite (TCL)** assure que
     $$\hat{\theta} \approx \mathcal{N}(\theta, \text{Var}(\hat{\theta}))$$
     pour $n$ grand.

    4. **Intervalle de confiance (IC)**  
       En inversant la distribution de $\hat{\theta}$, on construit un intervalle aléatoire
       qui contient $\theta$ avec une probabilité (fréquentiste) $1 - \alpha$.

    5. **Test d'hypothèse**  
       On choisit une hypothèse nulle $H_0$ et une alternative $H_1$, puis une statistique de test.
       - On calcule la **p-value** : probabilité (sous $H_0$) d’observer une valeur de la statistique 
     au moins aussi extrême que celle observée.
       - Si la p-value est plus petite que $\alpha$, on **rejette $H_0$**.

    Cette application montre comment **intervalle de confiance** et **test** sont deux faces d’un même objet :
    un paramètre est *rejeté* par un test bilatéral au niveau $\alpha$ si et seulement si **il ne
    se trouve pas** dans l’intervalle de confiance à $1-\alpha$.
    """
    )
    theorie
    return


if __name__ == "__main__":
    app.run()
