Here is a concise, math-focused description of each of the three zero-shot scoring strategies implemented in compute_fitness.py.

1) WT-marginals
We compute, for each position i, the log-probabilities
$$
\ell_i(a) = \log p\big(a \mid x_1, \dots, x_{i-1}, x_{i+1}, \dots, x_L\big)
$$
where $(x_1, \dots, x_L)$ is the wild-type sequence (including BOS/EOS tokens).

For a single mutation $(x_i = A \to B)$, the mutation score is
$$
s_i = \ell_i(B) - \ell_i(A).
$$
If there are multiple mutations $\{(i_k, A_k \to B_k)\}$, we sum:
$$
S = \sum_k \big[\ell_{i_k}(B_k) - \ell_{i_k}(A_k)\big].
$$

Handling very long sequences via overlapping windows
When $L > 1024$, we slide two 1024-long windows (from the left and right ends, stride = 511) and optionally a central window. Within each window $W$ we compute window-specific logits and log-softmaxes $\ell^{(W)}_{i}(a)$, then weight them by a position-dependent vector $w_j \in (0, 1]$ (smooth ramp up/down at the edges). Finally, for each $i$:
$$
\bar{\ell}_i(a) = \frac{\sum_{W \ni i} w_{i - W_{\rm start}} \, \ell^{(W)}_{i}(a)}{\sum_{W \ni i} w_{i - W_{\rm start}}}.
$$
These $\bar{\ell}_i(a)$ replace the single-window $\ell_i(a)$ in the formula above.

2) Masked-marginals
For each position $i$ we replace $x_i$ by the special mask token, call the resulting sequence $x^{\setminus i}$. We then compute
$$
\ell_i(a) = \log p\big(a \mid x^{\setminus i}\big).
$$
Scoring is again
$$
s_i = \ell_i(B) - \ell_i(A), \qquad S = \sum_i s_i,
$$
where $A \to B$ is the mutation at $i$. Long-sequence handling (“optimal” window) simply extracts the smallest 1024-long subsequence that still covers position $i$.

3) Pseudo-perplexity (“pseudo-ppl”)
We first generate the mutated sequence $y$ by applying all mutations to the wild-type. Then for each internal position $i$ we mask $y_i$ and compute
$$
\ell_i = \log p\big(y_i \mid y^{\setminus i}\big).
$$
The pseudo–log-likelihood of $y$ is
$$
\mathcal{L}(y) = \sum_{i=2}^{L-1} \ell_i = \sum_{i=2}^{L-1} \log p\big(y_i \mid y^{\setminus i}\big).
$$
This sum of log-probabilities is used directly as the fitness score (higher means more “likely” under the model).