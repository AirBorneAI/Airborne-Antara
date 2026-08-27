# 🧮 Mathematical Proof: Relational Graph Memory

**[⬅ Return to Architecture](../technical/SYNTHETIC_INTUITION.md) | [See Implementation (Code)](../../airborne_antara/memory.py)**

---

## 1. Graph Definitions

We define memory not as a queue, but as a directed graph $G_t = (V_t, E_t)$ at time $t$.
*   $V_t = \{n_1, ..., n_k\}$: The set of memory nodes. Each node $n_i$ contains a key vector $k_i \in \mathbb{R}^d$ and value $v_i$.
*   $E_t$: The set of edges representing semantic associations.

## 2. Associative Retrieval

Given a query vector $q_t$ (the current state), we compute the similarity score $\phi$ for every node $n_i$:

$$ \phi(q_t, k_i) = \text{CosineSimilarity}(q_t, k_i) = \frac{q_t \cdot k_i}{||q_t|| ||k_i||} $$

We retrieve the set of active memories $M_{active}$ where $\phi > \tau$ (activation threshold):

$$ M_{active} = \{ v_i \mid n_i \in V_t, \phi(q_t, k_i) > \tau \} $$

## 3. Spreading Activation (The "Relational" Step)

Standard vector databases stop at retrieval. AirborneHRS implements **Spreading Activation**.
If a node $n_i$ is activated, it propagates energy to its neighbors $n_j$ via edges $e_{ij} \in E_t$.

Energy $A_j$ for neighbor node $j$:
$$ A_j = \sum_{i \in M_{active}} w_{ij} \cdot \phi(q_t, k_i) \cdot \gamma $$

Where:
*   $w_{ij}$: Edge weight (strength of association).
*   $\gamma$: Decay factor (e.g., 0.5).

This allows the system to recall context that is not directly similar to the query but is *semantically linked* to the retrieved items.

## 4. Graph Construction (Learning)

When a new memory $n_{new}$ is added:
1.  **Node Creation**: $V_{t+1} = V_t \cup \{n_{new}\}$.
2.  **Edge Formation**: We compute similarity between $n_{new}$ and recent/active nodes $n_{recent}$. An edge is created if:

    $$ \phi(k_{new}, k_{recent}) > \tau_{link} $$

    The weight $w_{new, recent}$ is initialized to $\phi$.

## 5. Forgetting (Pruning)

To keep the graph bounded, we apply a **Forget Gate**. A node $n_i$ is pruned if its utility $u_i$ drops below a threshold:

$$ u_i(t) = \beta u_i(t-1) + (1-\beta) \cdot \mathbb{I}(n_i \in M_{active}) $$

Nodes that are rarely retrieved ($u_i \rightarrow 0$) are removed from $V$.
