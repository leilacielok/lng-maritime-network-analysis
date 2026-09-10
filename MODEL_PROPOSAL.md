# Preliminary Bayesian Model for LNG Node Criticality

## Status and purpose

This document summarises a preliminary modelling proposal for studying the criticality of LNG terminals and maritime chokepoints.
The exploratory analysis suggests that node criticality is multidimensional and that terminals and chokepoints should not be treated as statistically identical units. The following proposal therefore uses separate measurement models for the two node types while retaining the complete directed LNG network to account for network dependence, temporal dynamics, and common monthly shocks.
## 1. Dimensions of criticality

The proposed latent dimensions are:

1. **Exposure (E)**, provisionally represented by LNG throughput and voyage or route exposure, with degree considered as an additional connectivity indicator (the structure reflects the EDA, since throughput, strength, voyage exposure, route exposure and degree are found to be strongly correlated and should not be treated as independent evidence of criticality).
2. **Structural position (S)**, currently represented only by role-specific PageRank for terminals and weighted betweenness for chokepoints.
At present, structural position has only one preferred indicator for each node type, so we either add other appropriate structural indicators or we use transformed and standardised PageRank and betweenness directly as observed dynamic structural scores.

3.  **Dependence (D)**, defined only for terminals and represented by counterparty concentration (HHI) and role-/country-specific dependence measures.

## 2. Synthetic model specification

Let $i$ denote a terminal, $c$ a chokepoint, $t=1,\ldots,60$ a month, and $g\in{T,C}$ the node type.

### Terminal measurement model

$$
\mathbf y_{it}^{T*}
=
\boldsymbol\alpha^T
+
\boldsymbol\Lambda^T
\begin{pmatrix}
f_{it}^{E,T}\\
f_{it}^{S,T}\\
f_{it}^{D,T}
\end{pmatrix}
+
\boldsymbol\varepsilon_{it}^T.
$$

### Chokepoint measurement model

$$
\mathbf y_{ct}^{C*}
=
\boldsymbol\alpha^C
+
\boldsymbol\Lambda^C
\begin{pmatrix}
f_{ct}^{E,C}\\
f_{ct}^{S,C}
\end{pmatrix}
+
\boldsymbol\varepsilon_{ct}^C.
$$

### Dynamic network model for exposure

$$
f_{it}^{E,g}
=
\mu_E^g
+u_i^{E,g}
+\phi_E^g f_{i,t-1}^{E,g}
+\rho_E^g\sum_{\ell=1}^{187}
w_{\ell i,t-1}f_{\ell,t-1}^{E,g(\ell)}
+\delta_t^E
+\eta_{it}^{E,g}.
$$

The network term is introduced only with the exposure factor equation simply because exposure is the dimension most directly comparable across terminals and chokepoints. Network effects for the other dimensions can be tested  after finding other common indicators.

## 3. Measurement models

### 3.1 Terminals

The terminal factors are:

- $f_{it}^{E,T}$: exposure;
- $f_{it}^{S,T}$: structural position;
- $f_{it}^{D,T}$: dependence.

By considering as possible observed indicators throughput, voyage count, role-specific PageRank, counterparty HHI and role-/country-specific dependence, a eliminary confirmatory loadings matrix could be:

$$
\boldsymbol\Lambda^T=
\begin{pmatrix}
\lambda_{\mathrm{thr},E}^T & 0 & 0\\
\lambda_{\mathrm{voy},E}^T & 0 & 0\\
0 & \lambda_{\mathrm{PR},S}^T & 0\\
0 & 0 & \lambda_{\mathrm{HHI},D}^T\\
0 & 0 & \lambda_{\mathrm{dep},D}^T
\end{pmatrix}.
$$

### 3.2 Chokepoints

The chokepoint factors are:

- $f_{ct}^{E,C}$:scale and exposure;
- $f_{ct}^{S,C}$:structural position.

Following the same reasoning, when possible observed indicators are throughput, voyage exposure, route exposure and weighted betweenness, preliminary loading structure may appear as:

$$
\boldsymbol\Lambda^C=
\begin{pmatrix}
\lambda_{\mathrm{thr},E}^C & 0\\
\lambda_{\mathrm{voy},E}^C & 0\\
\lambda_{\mathrm{route},E}^C & 0\\
0 & \lambda_{\mathrm{bet},S}^C
\end{pmatrix}.
$$

### 3.3 Observation distributions

Strongly right-skewed non-negative indicators would need to be standardized as

$$
y_{itj}^*=\operatorname{standardize}\!\left[\log(1+y_{itj})\right].
$$

Their distribution would be conditional on the latent factors:

$$
y_{itj}^{g*}\sim
\mathcal N\!\left(
\alpha_j^g+{\boldsymbol\lambda_j^g}^{\!\top}\mathbf f_{it}^g,
(\sigma_j^g)^2
\right),
$$

with, for example,

$$
\alpha_j^g\sim\mathcal N(0,2^2),
\qquad
\lambda_{jk}^g\sim\operatorname{HalfNormal}(0,1),
\qquad
\sigma_j^g\sim\operatorname{HalfNormal}(0,1).
$$

## 4. Persistent, temporal and network components

### Node-specific persistence
$$
u_i^{E,g}\sim\mathcal N(0,\tau_{E,g}^2),
\qquad
\tau_{E,g}\sim\operatorname{HalfNormal}(0,1).
$$

The term $u_i^{E,g}$ captures the persistent tendency of a node to have exposure above or below the average for its type. 

### Temporal persistence

The introduction of an autoregressive component is motivated primarily by the positive month-to-month rank persistence observed in the EDA. e coefficient $\phi_{E}^g$ measures how strongly a node's current exposure depends on its own exposure in the preceding month. A possible stationary prior is

$$
\phi_E^g\sim\mathcal N(0,0.5^2)\;\mathbb I(-1<\phi_E^g<1).
$$

### Network dependence

The maweighted adjency trix (W${t-1}=[w_{\ell i,t-1}]) $  obtained from the complete directed, flow-weighted LNG network and is row-normalised. T The direction of $W$ must be stated explicitly, and incoming and outgoing normalisations may be compared in sensitivity analyses because they represent different economic mechanisms.
 term

$$
\sum_{\ell=1}^{187}w_{\ell i,t-1}f_{\ell,t-1}^{E,g(\ell)}
$$

is a flow-weighted summary of the previous exposure of nodes connected to node `i$.$, w ile thcoefficient `r$\o_E^g` $easures network dependence a, that is how much the previous exposure of neighbours is associted to the present exposure of node $i$An initial shrinkage prior may be

$$
\rho_E^g\sim\mathcal N(0,0.3^2).
$$

UsUsing the lagged network matrix $W_{t-1}$ establishes a temporal ordering: the network connections and node exposures observed in month $t-1$ are used to model node exposure in month $t$. This also reduces the risk of a mechanical association that could arise if both the network weights and the exposure indicators were constructed from the same contemporaneous LNG flows.## Node-month innovation

$$
\eta_{it}^{E,g}\sim\mathcal N(0,\sigma_{\eta,E,g}^2).
$$

This is the remaining node-specific monthly variation after accounting for persistent heterogeneity, temporal persistence, network dependence and the common monthly shock.

## 5. Common monthly shock

ThAs a first modelling proposal, tsame monthly effect enters the exposure equation of every node:

$$
\delta_t^E
=
\gamma_E\delta_{t-1}^E
+
\zeta_t^E,
\qquad
\zeta_t^E\overset{\mathrm{iid}}{\sim}
\mathcal N(0,\sigma_{\delta,E}^2).
$$

Here:

- $\delta_t^E$ is the common state of the LNG system in month $t$
- $\gamma_E$ masures the persistence of that common state;
- $\zeta_t^E$ is the new system-wide innovation occurring in month $t$
- $\sigma_{\delta,E}$ determines the typical size of new common shocks.

ForStationarity ($-1<\gamma_E<1$) is adopted as an initial simplifying assumption: common monthly shocks may persist, but their effects are assumed to diminish over time rather than accumulate permanently. Nevertheless, this assumption will need to be assessed through posterior estimates and sensitivity analyses. ossible prior specification is

$$
\gamma_E\sim\mathcal N(0,0.5^2)\;\mathbb I(-1<\gamma_E<1),
\qquad
\sigma_{\delta,E}\sim\operatorname{HalfNormal}(0,0.5).
$$

The stationary initial distribution is

$$
\delta_1^E\sim
\mathcal N\!\left(
0,
\frac{\sigma_{\delta,E}^2}{1-\gamma_E^2}
\right).
$$

The distinction from the node-specific innovation is important: $\zeta_t^E$ varies by month but not by node, whereas $\eta_{it}^{E,g}$ varies by both node and month. The common effect may represent genuine market-wide variation, but it may also absorb changes in data coverage (the 13 unusually low-activity months identified by the QA should therefore be examined through sensitivity analyses).

## 6. Dynamics of the other dimensions

For an initial model, we can use simpler dynamic equations without a network lag:

$$
f_{it}^{S,g}
=
\mu_S^g+u_i^{S,g}
+\phi_S^g f_{i,t-1}^{S,g}
+\delta_t^{S,g}+\eta_{it}^{S,g},
$$

and, for terminals only,

$$
f_{it}^{D,T}
=
\mu_D^T+u_i^{D,T}
+\phi_D^T f_{i,t-1}^{D,T}
+\delta_t^{D,T}+\eta_{it}^{D,T}.
$$

## 7. Inactive node-months

The balanced panels contain many inactive observations, so it would be probably better to aknowledge activity:

$$
A_{it}^g\sim\operatorname{Bernoulli}(p_{it}^g),
$$

$$
\operatorname{logit}(p_{it}^g)
=a_g+b_i^{A,g}+\psi_g A_{i,t-1}^g+\delta_t^{A,g}.
$$

The continuous measurement model would then be estimated conditional on $A_{it}^g=1$. This hurdle formulation separates the probability of being active from the level of criticality conditional on activity. For an initial implementation, we also may implement the model just on active node-months.

