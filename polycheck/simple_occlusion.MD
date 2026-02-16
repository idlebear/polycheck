# Simplified Occlusion (For Single Cell Dynamic Occluders + Bresenham LOS)

This note specifies a **stochastic occlusion/visibility model** for tracking a target of interest (ToI) on a grid when **dynamic occlusions** are induced by other independently moving agents (“occluders”). The goal is to compute, at each time step, the **probability that a given sensor can observe a given target cell**, and then the induced probability of obtaining a non-occluded measurement.

## 1. State, agents, and independence assumptions

- Let \(X_t \in \mathcal S\) be the ToI (tracked target) position at time \(t\).
- Let \(Z_t^{(j)} \in \mathcal S\) be the position of occluder \(j\in\{1,\dots,J\}\) at time \(t\).
- Each occluder follows its own Markov chain (possibly different per agent). **Occluders are assumed independent across \(j\)**.

Let the predicted occupancy belief for occluder \(j\) at time \(t\) be:

\[
b_t^{(j)}(c) \triangleq \Pr\big(Z_t^{(j)} = c\big), \qquad c\in\mathcal S.
\]

Let the predicted occupancy belief for the ToI be:

\[
b_t(x) \triangleq \Pr\big(X_t = x\big), \qquad x\in\mathcal S.
\]

(These may be the result of filtering and/or one-step/multi-step prediction.)

## 2. Bresenham LOS and blocking-cell sets

For each sensor \(m\) and each candidate target cell \(x\in\mathcal S\), compute the discrete line-of-sight ray using Bresenham’s algorithm:

- \(\texttt{Ray}(m,x)\): ordered list of grid cells from the sensor location to \(x\).

Define the **blocking set** as the intervening cells along the ray:

\[
C(m,x) \triangleq \texttt{Ray}(m,x)\setminus\{x\}.
\]

Optionally, if you prefer “strictly between,” you may also remove the sensor’s own cell from \(C(m,x)\).

> **Interpretation:** An occluder blocks visibility to \(x\) from sensor \(m\) if it occupies **any** cell in \(C(m,x)\).

## 3. Exact per-occluder LOS blocking probability (single-cell occupancy)

Under the standard grid assumption that each occluder occupies **exactly one cell** at any time step, the events \(\{Z_t^{(j)}=c\}\) are mutually exclusive across cells \(c\).

Therefore, for a fixed \(m\), \(x\), and \(t\), the probability that occluder \(j\) blocks the LOS to \(x\) is:

\[
p_{\text{block}}^{(j)}(m,x,t)
\;\triangleq\;
\Pr\big(Z_t^{(j)}\in C(m,x)\big)
\;=\;
\sum_{c\in C(m,x)} b_t^{(j)}(c).
\]

The probability that occluder \(j\) **does not** block the LOS is:

\[
p_{\text{clear}}^{(j)}(m,x,t)
\;=\;
1 - \sum_{c\in C(m,x)} b_t^{(j)}(c).
\]

> This formulation captures **between-cell** occlusion (occluder on the ray), not just target/occluder overlap.

## 4. Combining multiple independent occluders

Assuming occluders are independent across \(j\), the LOS is clear if **all** occluders are clear:

\[
p_{\text{clear}}(m,x,t)
\;=\;
\prod_{j=1}^{J} p_{\text{clear}}^{(j)}(m,x,t)
\;=\;
\prod_{j=1}^{J}\left(1 - \sum_{c\in C(m,x)} b_t^{(j)}(c)\right).
\]

Then the geometric visibility probability (ignoring sensor miss) is:

\[
p_{\text{vis}}(x; m, t) \triangleq \mathbf{1}\{x\in\mathcal S_m\}\; p_{\text{clear}}(m,x,t),
\]

where \(\mathcal S_m\) is the coverage set of sensor \(m\).

## 5. Incorporating measurement miss probability

If the sensor returns an observation of the target with probability \(1-p_{\textsc{miss}}\) when visible, then:

\[
\Pr(O_t=1\mid X_t=x, A_t=\{m\})
\;=\;
(1-p_{\textsc{miss}})\; p_{\text{vis}}(x; m, t),
\]

where \(O_t\in\{0,1\}\) indicates whether a non-occluded (state-valued) observation was obtained (e.g., \(Y_t\neq\mathtt{occ}\)).

## 6. Probability of observing the ToI under a sensor choice

Given the ToI belief \(b_t(x)\), the probability that sensor \(m\) yields a non-occluded observation at time \(t\) is:

\[
p_{\text{obs}}(m,t)
\;\triangleq\;
\Pr(O_t=1\mid A_t=\{m\})
\;=\;
(1-p_{\textsc{miss}})\sum_{x\in\mathcal S} b_t(x)\; p_{\text{vis}}(x;m,t).
\]

For a sensor subset \(A_t\), a conservative combination is:

\[
p_{\text{vis}}(x;A_t,t) \approx \max_{m\in A_t} p_{\text{vis}}(x;m,t),
\]

or, if you are comfortable with an independence approximation across sensors,

\[
p_{\text{vis}}(x;A_t,t) \approx 1-\prod_{m\in A_t}\bigl(1-p_{\text{vis}}(x;m,t)\bigr).
\]

## 7. Notes on the earlier “product over cells” heuristic

A sometimes-used heuristic for a single occluder is:

\[
\prod_{c\in C(m,x)} \bigl(1-b_t^{(j)}(c)\bigr),
\]

which treats cell occupancies as independent Bernoulli events for a single agent. This conflicts with the **single-cell occupancy** constraint (one agent cannot occupy multiple cells simultaneously). The **sum-over-ray** expression in Section 3 is the correct “exact” form under single-cell occupancy.

## 8. GPU implementation sketch (Bresenham rays)

Precompute the blocking sets \(C(m,x)\) using Bresenham and store them in a ragged format (e.g., CSR):

- `ray_ptr[m,x]`: start offset in a flattened array
- `ray_len[m,x]`: number of blocking cells
- `ray_cells[...]`: flattened list of cell indices on the ray (excluding \(x\))

Per time step:

1. For each occluder \(j\) and each \((m,x)\), compute the segmented sum
   \[
   s_j(m,x,t) = \sum_{c\in C(m,x)} b_t^{(j)}(c).
   \]
2. Compute \(p_{\text{clear}}(m,x,t)=\prod_j (1-s_j(m,x,t))\) (often in log space for stability).
3. Weight by \(b_t(x)\) and sum over \(x\) to get \(p_{\text{obs}}(m,t)\).

---

**End of document.**
