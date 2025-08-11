# Universal Low-Parameter Ratio Preserving Adaptor

An efficient method for constructing **low-parameter linear transformations** between high-dimensional feature spaces. Unlike typical low-parameter adaptation methods that create information bottlenecks through a single reduced dimension, our approach maintains full feature access through parallel pathways. The method guarantees:

- **Preservation of the global input/output ratio**
- **No bottlenecks or chokepoints** that could limit information flow
- **High GCD alignment** in sub-mappings for efficient reshaping and computation

## Problem Statement

Given input and output dimensions $x > 0$, $y > 0$, we aim to construct a mapping from $\mathbb{R}^x \to \mathbb{R}^y$ with reduced parameter complexity and guaranteed structural properties. The method decomposes the full mapping into two blocks:

$$(x_1, y_1), \quad (x_2, y_2)$$

such that:

1. $x_1 + x_2 = x$, $y_1 + y_2 = y$
2. Each local ratio $\frac{x_i}{y_i}$ approximates the global ratio $\frac{x}{y}$
3. $\gcd(x_i, y_i)$ is large for efficient reshaping and grouped computation
4. Each chunk supports low-parameter mapping with full feature access (no chokepoints)

## Method

Let:

$$r = \frac{x}{y}, \quad r_{\text{near}} = \text{round}(r)$$

To decompose the mapping, solve for $n \in \mathbb{Z}$ in the equation:

$$\frac{x - r_{\text{near}} \cdot n}{y - n} = r_{\text{alt}}$$

where $r_{\text{alt}} \in \{ \lfloor r \rfloor, \lceil r \rceil \} \setminus \{ r_{\text{near}} \}$.

Solving yields:

$$n = \frac{x - r_{\text{alt}} \cdot y}{r_{\text{near}} - r_{\text{alt}}}$$

This gives the block dimensions:

- $y_1 = n$, $x_1 = r_{\text{near}} \cdot n$
- $y_2 = y - n$, $x_2 = x - x_1$

Each chunk can now be handled independently, maintaining strong alignment with the global ratio while enabling efficient projection structures.

## Example

Let $x = 16000$, $y = 10000$. Then:

$$r = 1.6, \quad r_{\text{near}} = 2, \quad r_{\text{alt}} = 1$$

Solve:

$$n = \frac{16000 - 1 \cdot 10000}{2 - 1} = 6000$$

Then:

- Block 1: $x_1 = 2 \cdot 6000 = 12000$, $y_1 = 6000$
- Block 2: $x_2 = 4000$, $y_2 = 4000$

This produces two perfectly aligned blocks:
- One with ratio $2:1$
- One with ratio $1:1$

Each can be reshaped into groups of size $\gcd(x_i, y_i)$, then projected with a shared or low-parameter matrix of shape $g \to 1$.

## Parameter Efficiency

The key insight is that we can bound the total number of parameters needed for adaptation. For any input dimension $x$ and output dimension $y$, the maximum parameter ratio is bounded by:

$$2 \cdot \max(\frac{x}{y}, \frac{y}{x})$$

We achieve this efficiency by:
1. Splitting the tensor into blocks that preserve the input/output ratio
2. Utilizing the greatest common factor (GCD) between current and target dimensions in each block

For each block $i$, we identify groups of size $g_i = \gcd(x_i, y_i)$. This allows us to reshape the tensor into $\frac{x_i}{g_i}$ groups of size $g_i$, each mapped to $\frac{y_i}{g_i}$ outputs. The total parameter count becomes:

$$\sum_{i=1}^2 \left( \frac{x_i}{g_i} \cdot \text{rank}(g_i \to 1) \right)$$

This approach is significantly more efficient than a naive dense projection, especially for dimensions with large common factors. For example, when adapting between multiples of a base dimension (like 768 ↔ 1024), the GCD-based grouping can reduce parameters by orders of magnitude while maintaining full expressivity.