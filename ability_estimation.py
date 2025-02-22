import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import norm
import plotly.io as pio
pio.templates.default = "ggplot2"

# -------------------------------------------------
# General IRT Log-Likelihood Function
# -------------------------------------------------
def irt_log_likelihood(theta, responses, difficulties, discriminations=None, guesses=None):
    """
    Computes the log-likelihood for binary responses under a general IRT model.
    
    For the 1PL model, discriminations default to 1 and guessing to 0.
    For the 2PL model, supply discriminations; for the 3PL model, supply both discriminations and guesses.
    """
    difficulties = np.array(difficulties)
    if discriminations is None:
        discriminations = np.ones_like(difficulties)
    else:
        discriminations = np.array(discriminations)
    if guesses is None:
        guesses = np.zeros_like(difficulties)
    else:
        guesses = np.array(guesses)
    
    exp_term = np.exp(discriminations * (theta - difficulties))
    logistic = exp_term / (1 + exp_term)
    p = guesses + (1 - guesses) * logistic
    eps = 1e-10
    p = np.clip(p, eps, 1 - eps)
    log_like = np.sum(responses * np.log(p) + (1 - responses) * np.log(1 - p))
    return log_like

# -------------------------------------------------
# EAP Estimation Using Gauss–Legendre Quadrature (Uniform Prior)
# -------------------------------------------------
def eap_estimate_uniform_prior_quad(responses, difficulties, discriminations=None, guesses=None,
                                    lower=-3, upper=3, n_nodes=64):
    """
    Estimate theta using the Expected A Posteriori (EAP) method with a uniform prior on [lower, upper]
    via Gauss–Legendre quadrature.
    """
    nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
    theta_nodes = 0.5 * (upper - lower) * nodes + 0.5 * (upper + lower)
    weights = 0.5 * (upper - lower) * weights

    log_likelihood = np.array([
        irt_log_likelihood(theta, responses, difficulties, discriminations, guesses)
        for theta in theta_nodes
    ])
    likelihood = np.exp(log_likelihood)
    posterior_unnorm = likelihood
    denom = np.sum(weights * posterior_unnorm)
    numer = np.sum(weights * theta_nodes * posterior_unnorm)
    theta_eap = numer / denom
    variance = np.sum(weights * (theta_nodes - theta_eap)**2 * posterior_unnorm) / denom
    se_eap = np.sqrt(variance)
    return theta_eap, se_eap, theta_nodes, posterior_unnorm

# -------------------------------------------------
# Test Information Function
# -------------------------------------------------
def test_information(thetas, difficulties, discriminations=None, guesses=None):
    """
    Computes the test information function as the sum of item information over items.
    
    $$ I_i(\theta)=\frac{\left[(1-c_i)a_iL(\theta-\delta_i)(1-L(\theta-\delta_i))\right]^2}{p_i(\theta)(1-p_i(\theta))}, $$
    
    where
    
    $$ L(\theta-\delta_i)=\frac{\exp\{a_i(\theta-\delta_i)\}}{1+\exp\{a_i(\theta-\delta_i)\}}, $$
    
    and
    
    $$ p_i(\theta)=c_i+(1-c_i)L(\theta-\delta_i). $$
    """
    difficulties = np.array(difficulties)
    if discriminations is None:
        discriminations = np.ones_like(difficulties)
    else:
        discriminations = np.array(discriminations)
    if guesses is None:
        guesses = np.zeros_like(difficulties)
    else:
        guesses = np.array(guesses)
    
    info = []
    for theta in thetas:
        item_infos = []
        for d, a, c in zip(difficulties, discriminations, guesses):
            L = np.exp(a*(theta-d))/(1+np.exp(a*(theta-d)))
            p = c + (1-c)*L
            dp = (1-c)*a*L*(1-L)
            item_info = (dp**2) / (p*(1-p))
            item_infos.append(item_info)
        info.append(np.sum(item_infos))
    return np.array(info)

# -------------------------------------------------
# ICC (Item Characteristic Curve) Function
# -------------------------------------------------
def compute_icc(theta, difficulty, discrimination=1, guessing=0):
    exp_term = np.exp(discrimination * (theta - difficulty))
    logistic = exp_term / (1 + exp_term)
    p = guessing + (1 - guessing) * logistic
    return p

# -------------------------------------------------
# Item Information Curve (IIC) Function for a Single Item
# -------------------------------------------------
def compute_item_information(theta, difficulty, discrimination=1, guessing=0):
    L = np.exp(discrimination*(theta - difficulty))/(1+np.exp(discrimination*(theta - difficulty)))
    p = guessing + (1-guessing)*L
    dp = (1-guessing)*discrimination * L * (1 - L)
    item_info = (dp**2) / (p*(1-p))
    return item_info

# -------------------------------------------------
# Marginal Reliability Calculation Using Gaussian Quadrature
# -------------------------------------------------
def marginal_reliability_quad(difficulties, discriminations, guesses, lower=-3, upper=3, n_nodes=64):
    """
    Compute marginal reliability via Gauss–Legendre quadrature.
    Conditional reliability is:
    
    $$ r(\theta)=\frac{I(\theta)}{I(\theta)+1}, $$
    
    and marginal reliability is computed as:
    
    $$ \text{Marginal Reliability} = \frac{\int_{-3}^{3} r(\theta)\,\phi(\theta)\,d\theta}{\int_{-3}^{3} \phi(\theta)\,d\theta}, $$
    
    where $\phi(\theta)$ is the standard normal density.
    """
    nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
    theta_nodes = 0.5 * (upper - lower) * nodes + 0.5 * (upper + lower)
    weights = 0.5 * (upper - lower) * weights
    I_nodes = test_information(theta_nodes, difficulties, discriminations, guesses)
    r_theta = I_nodes / (I_nodes + 1)
    f_theta = norm.pdf(theta_nodes)
    marginal_r = np.sum(weights * r_theta * f_theta) / np.sum(weights * f_theta)
    return marginal_r

# -------------------------------------------------
# Streamlit App Interface
# -------------------------------------------------
st.title("Estimating Person Ability in IRT - A Quick Tour")

st.write('This demo estimates person ability parameters (assuming known item parameters) using EAP with a uniform prior (θ ~ [-3,3]) under different IRT models. The app also shows the Test Information Function, marginal reliability, the Item Characteristic Curve (ICC), and the Item Information Curve (IIC) for a selected item. For questions, please contact: <jianganghao@gmail.com>')

with st.expander("Click here to see the Math"):
    st.markdown(r"""
    #### IRT Models

    In Item Response Theory (IRT), the probability that an examinee with ability $\theta$ responds correctly to item $i$ is modeled differently depending on the model:

    - **1PL Model (Rasch Model):**

      $$
      p_i(\theta) = \frac{\exp(\theta-\delta_i)}{1+\exp(\theta-\delta_i)}.
      $$
      
      Here, discrimination is 1 and guessing is 0.

    - **2PL Model:**

      $$
      p_i(\theta) = \frac{\exp\{a_i(\theta-\delta_i)\}}{1+\exp\{a_i(\theta-\delta_i)\}}.
      $$
      
      The guessing parameter is assumed to be 0.

    - **3PL Model:**

      $$
      p_i(\theta) = c_i+(1-c_i)\frac{\exp\{a_i(\theta-\delta_i)\}}{1+\exp\{a_i(\theta-\delta_i)\}}.
      $$

    #### Log-Likelihood

    $$
    \ell(\theta) = \sum_{i=1}^n \left[ x_i\ln(p_i(\theta)) + (1-x_i)\ln(1-p_i(\theta)) \right].
    $$

    #### Expected A Posteriori (EAP) Estimation

    $$
    \hat{\theta}_{EAP} = \frac{\int \theta \, L(\theta)\,\pi(\theta)\,d\theta}{\int L(\theta)\,\pi(\theta)\,d\theta},
    $$
    
    where $L(\theta)=\exp\{\ell(\theta)\}$.

    #### Gaussian Quadrature for Numerical Integration

    $$
    \int_a^b f(\theta)\,d\theta \approx \sum_{j=1}^N w_j\, f(\theta_j).
    $$

    #### Test Information Function

    $$
    I(\theta)=\sum_{i=1}^n I_i(\theta),
    $$
    
    where
    
    $$
    I_i(\theta)=\frac{\left[(1-c_i)a_iL(\theta-\delta_i)(1-L(\theta-\delta_i))\right]^2}{p_i(\theta)(1-p_i(\theta))},
    $$
    
    with
    
    $$
    L(\theta-\delta_i)=\frac{\exp\{a_i(\theta-\delta_i)\}}{1+\exp\{a_i(\theta-\delta_i)\}},
    $$
    
    and
    
    $$
    p_i(\theta)=c_i+(1-c_i)L(\theta-\delta_i).
    $$

    #### Marginal Reliability

    Conditional reliability is defined as:
    
    $$
    r(\theta)=\frac{I(\theta)}{I(\theta)+1}.
    $$
    
    Marginal reliability is computed as the weighted average of $r(\theta)$ using the standard normal density:
    
    $$
    \text{Marginal Reliability} = \frac{\int_{-3}^{3} r(\theta)\,\phi(\theta)\,d\theta}{\int_{-3}^{3} \phi(\theta)\,d\theta},
    $$
    
    where $\phi(\theta)$ is the standard normal density.

    #### Item Characteristic Curve (ICC)

    The ICC for item $i$ is the function $p_i(\theta)$.

    #### Summary

    - **1PL Model:**
      $$
      p_i(\theta)=\frac{\exp(\theta-\delta_i)}{1+\exp(\theta-\delta_i)}.
      $$
      
    - **2PL Model:**
      $$
      p_i(\theta)=\frac{\exp\{a_i(\theta-\delta_i)\}}{1+\exp\{a_i(\theta-\delta_i)\}}.
      $$
      
    - **3PL Model:**
      $$
      p_i(\theta)=c_i+(1-c_i)\frac{\exp\{a_i(\theta-\delta_i)\}}{1+\exp\{a_i(\theta-\delta_i)\}}.
      $$
      
    - **Log-Likelihood:**
      $$
      \ell(\theta)=\sum_{i=1}^n \left[x_i\ln(p_i(\theta))+(1-x_i)\ln(1-p_i(\theta))\right].
      $$
      
    - **EAP Estimate:**
      $$
      \hat{\theta}_{EAP}=\frac{\int \theta\, L(\theta)\,\pi(\theta)\,d\theta}{\int L(\theta)\,\pi(\theta)\,d\theta}.
      $$
      
    - **Gaussian Quadrature:**
      $$
      \int_a^b f(\theta)\,d\theta\approx \sum_{j=1}^N w_j\, f(\theta_j).
      $$
    """)
    
st.markdown('---------')

# -------------------------------------------------
# Main Controls (No Sidebar)
# -------------------------------------------------
st.markdown("### Simulation Controls")
model_type = st.radio("Select IRT Model", options=["1PL", "2PL", "3PL"], horizontal=True)
n_items = st.slider("Number of Items", min_value=1, max_value=300, value=30, step=1)

# -------------------------------------------------
# Parameter Error Settings
# -------------------------------------------------
#st.markdown("##### Setting Parameter Estimation Error level")
difficulty_error_level = st.slider("Difficulty Parameter Error Level (std dev)", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
if model_type in ["2PL", "3PL"]:
    discrimination_error_level = st.slider("Discrimination Parameter Error Level (std dev)", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
else:
    discrimination_error_level = 0.0
if model_type == "3PL":
    guessing_error_level = st.slider("Guessing Parameter Error Level (std dev)", min_value=0.0, max_value=0.5, value=0.0, step=0.05)
else:
    guessing_error_level = 0.0

# -------------------------------------------------
# Data Simulation and EAP Estimation
# -------------------------------------------------
np.random.seed(42)
# Generate base item difficulties from N(0,1)
base_difficulties = np.random.normal(0, 1, n_items)
# Add Gaussian error to difficulties based on selected error level
difficulties = base_difficulties + np.random.normal(0, difficulty_error_level, n_items)

# For 2PL and 3PL: generate base discrimination parameters from Uniform(0.8, 1.2)
if model_type in ["2PL", "3PL"]:
    base_discriminations = np.random.uniform(0.8, 1.2, n_items)
    # Add Gaussian error to discrimination
    discriminations = base_discriminations + np.random.normal(0, discrimination_error_level, n_items)
else:
    discriminations = None

# For 3PL: generate base guessing parameters from Uniform(0.1, 0.25)
if model_type == "3PL":
    base_guessing = np.random.uniform(0.1, 0.25, n_items)
    # Add Gaussian error to guessing
    guesses = base_guessing + np.random.normal(0, guessing_error_level, n_items)
    # Ensure guessing is between 0 and 1
    guesses = np.clip(guesses, 0, 1)
else:
    guesses = None


theta_true_values = np.arange(-3, 3.01, 0.25)
estimated_thetas = []
standard_errors = []

for true_theta in theta_true_values:
    if model_type == "1PL":
        p = np.exp(true_theta - base_difficulties) / (1 + np.exp(true_theta - base_difficulties))
    elif model_type == "2PL":
        exp_term = np.exp(base_discriminations * (true_theta - base_difficulties))
        p = exp_term / (1 + exp_term)
    elif model_type == "3PL":
        exp_term = np.exp(base_discriminations * (true_theta - base_difficulties))
        logistic = exp_term / (1 + exp_term)
        p = base_guessing + (1 - base_guessing) * logistic

    responses = np.random.binomial(1, p)
    theta_eap, se_eap, theta_nodes, posterior = eap_estimate_uniform_prior_quad(
        responses, difficulties, discriminations=discriminations, guesses=guesses
    )
    estimated_thetas.append(theta_eap)
    standard_errors.append(se_eap)

results_df = pd.DataFrame({
    "True Theta": theta_true_values,
    "Estimated Theta": estimated_thetas,
    "Standard Error": standard_errors
})
items_df = pd.DataFrame({
    "Item": [f"Item {i+1}" for i in range(n_items)],
    "Difficulty": difficulties
})
if model_type in ["2PL", "3PL"]:
    items_df["Discrimination"] = discriminations
if model_type == "3PL":
    items_df["Guessing"] = guesses

with st.expander("Show the data"):
    cols = st.columns(2)
    with cols[0]:
        st.write("### Estimation Results")
        st.dataframe(results_df, use_container_width=True)
    with cols[1]:
        st.write("### Item Parameters")
        st.dataframe(items_df, use_container_width=True)

# -------------------------------------------------
# Test Information Function Calculation
# -------------------------------------------------
theta_grid = np.arange(-3, 3.01, 0.25)
test_info_vals = test_information(theta_grid, difficulties, discriminations=discriminations, guesses=guesses)
info_df = pd.DataFrame({
    "Theta": theta_grid,
    "Test Information": test_info_vals
})

# Calculate marginal reliability using Gaussian quadrature (64 nodes)
marginal_r = marginal_reliability_quad(difficulties, discriminations, guesses, lower=-3, upper=3, n_nodes=64)

st.markdown("### Simulation Plots")
st.markdown(f"**Marginal Reliability:** {marginal_r:.3f}")

# -------------------------------------------------
# Plot 1: Estimated Theta vs. True Theta
# -------------------------------------------------
fig1 = px.scatter(results_df, x="True Theta", y="Estimated Theta", error_y="Standard Error",
                  title=f"EAP Estimation for {model_type} Model",
                  labels={"True Theta": "True Theta", "Estimated Theta": "Estimated Theta"})
fig1.add_shape(type="line",
               x0=results_df["True Theta"].min(), y0=results_df["True Theta"].min(),
               x1=results_df["True Theta"].max(), y1=results_df["True Theta"].max(),
               line=dict(color="red", dash="dash"),
               xref="x", yref="y")
fig1.update_xaxes(showgrid=True, gridcolor="rgba(0,100,0,0.5)", griddash="dash")
fig1.update_yaxes(showgrid=True, gridcolor="rgba(0,100,0,0.5)", griddash="dash")
st.plotly_chart(fig1)

# -------------------------------------------------
# Plot 2: Test Information Function
# -------------------------------------------------
fig2 = px.line(info_df, x="Theta", y="Test Information", title="Test Information Function",
               labels={"Theta": "Theta", "Test Information": "Test Information"})
fig2.update_xaxes(showgrid=True, gridcolor="rgba(0,100,0,0.5)", griddash="dash")
fig2.update_yaxes(showgrid=True, gridcolor="rgba(0,100,0,0.5)", griddash="dash")
st.plotly_chart(fig2)

# -------------------------------------------------
# Deep Dive into Items: ICC & IIC
# -------------------------------------------------
st.markdown("### Deep Dive into Items")
selected_item = st.selectbox("Select an Item for ICC & IIC Plots", options=[f"Item {i+1}" for i in range(n_items)])
selected_index = int(selected_item.split(" ")[1]) - 1

# Display selected item parameters
selected_params = {
    "Difficulty": difficulties[selected_index],
    "Discrimination": discriminations[selected_index] if discriminations is not None else 1,
    "Guessing": guesses[selected_index] if guesses is not None else 0
}
selected_params_df = pd.DataFrame(selected_params, index=[selected_item])
st.markdown("#### Selected Item Parameters")
st.table(selected_params_df)

# ICC Plot
theta_icc = np.arange(-3, 3.1, 0.1)
icc_values = compute_icc(theta_icc, difficulties[selected_index],
                         discrimination=(discriminations[selected_index] if discriminations is not None else 1),
                         guessing=(guesses[selected_index] if guesses is not None else 0))
icc_df = pd.DataFrame({
    "Theta": theta_icc,
    "ICC": icc_values
})
fig3 = px.line(icc_df, x="Theta", y="ICC", title=f"ICC for {selected_item}",
               labels={"Theta": "Theta", "ICC": "Probability of Correct Response"})
fig3.update_xaxes(showgrid=True, gridcolor="rgba(0,100,0,0.5)", griddash="dash")
fig3.update_yaxes(showgrid=True, gridcolor="rgba(0,100,0,0.5)", griddash="dash")
st.plotly_chart(fig3)

# IIC Plot
theta_iic = np.arange(-3, 3.1, 0.1)
item_info_vals = [compute_item_information(theta, 
                                             difficulties[selected_index],
                                             discrimination=(discriminations[selected_index] if discriminations is not None else 1),
                                             guessing=(guesses[selected_index] if guesses is not None else 0))
                  for theta in theta_iic]
iic_df = pd.DataFrame({
    "Theta": theta_iic,
    "Item Information": item_info_vals
})
fig4 = px.line(iic_df, x="Theta", y="Item Information", title=f"Item Information Curve for {selected_item}",
               labels={"Theta": "Theta", "Item Information": "Information"})
fig4.update_xaxes(showgrid=True, gridcolor="rgba(0,100,0,0.5)", griddash="dash")
fig4.update_yaxes(showgrid=True, gridcolor="rgba(0,100,0,0.5)", griddash="dash")
st.plotly_chart(fig4)
