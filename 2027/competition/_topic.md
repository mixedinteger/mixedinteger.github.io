## Topic

The topic of the 2027 MIP competition is "Explainability and Small Infeasible Subsystems". The goal
is to advance the state of the art on computing small infeasible subsystems.

### Motivation

Explaining the outcome of a MIP (for example, why a model is infeasible, or how it could be made
feasible) is a natural question that is hard to answer because of the combinatorial nature of the
problem.

A widely used feature of commercial MIP solvers for explaining infeasibility is the so-called
Irreducible Infeasible System (IIS). In the literature, an IIS is typically defined as a subset of
the original constraints of the MIP formulation (including variable-bound constraints and
integrality constraints) that is infeasible, but becomes feasible if any single element is removed.

Producing a small IIS directly helps provide a short, readable explanation of the infeasibility of a
potentially much larger model. Yet, while most modern commercial MIP solvers have developed
(heuristic) methods to compute IISs in the context of MIP, there is very little scientific
literature (see references below) on good techniques for computing IISs, and open-source
implementations are rare.

### Problem Definition

We are given a MIP of the form

$$
\begin{aligned}
\min \quad & c^T x\\
\text{s.t.} \quad & Ax\leq b \\
& x_i \in \mathbb{Z} && \forall i \in I, \\
& x \in \mathbb{R}^n.
\end{aligned}
$$

The objective function is ignored when determining feasibility.

Here $I \subseteq \{1,\ldots,n\}$ is the index set of the integer variables. Note that the system
$Ax \le b$ may include variable-bound constraints, i.e., constraints of the form $x_i \le u_i$ or
$x_i \ge l_i$ for some $i \in \{1,\ldots,n\}$. All linear constraints are indexed by the set $M$,
and $a_r^T x \le b_r$ denotes the constraint with index $r \in M$.

In this competition, a subsystem is obtained from the original MIP by:

1. removing a subset of linear constraints (including bound constraints), and/or
2. removing integrality constraints for a subset of integer variables.

Let $C \subseteq M$ be the set of kept linear constraints and let $J \subseteq I$ be the set of
variables that remain integer. The resulting subsystem is denoted by $MIP(C,J)$ and is formally
defined as

$$
\begin{aligned}
\min \quad & c^T x\\
\text{s.t.} \quad & a_r^T x \le b_r && \forall r \in C, \\
& x_j \in \mathbb{Z} && \forall j \in J, \\
& x \in \mathbb{R}^n.
\end{aligned}
$$

Following Guieu and Chinneck (1999), for an infeasible instance of MIP, we say that:

- $MIP(C,J)$ is an Infeasible System (IS) if it is also infeasible;
- furthermore, an IS is called an Irreducible Infeasible System (IIS) if:
  1. for every constraint $r \in C$, the subsystem $MIP(C \setminus \{r\}, J)$ is feasible;
  2. for every integer variable $j \in J$, the subsystem $MIP(C, J \setminus \{j\})$ is feasible.

This is a single-element irreducibility definition: each linear constraint is one element,
whatever its sense ($\le$, $\ge$ or $=$), and each lower or upper bound of a variable is an element
of its own. In particular, relaxing a variable bound is treated as removing the corresponding bound
constraint. Note, however, that bound constraints of a
variable that remains integer in the subsystem (i.e., $x_j$ with $j \in J$) are always considered
part of the subsystem: they are neither counted in its size nor considered for minimality.

The _size_ of $MIP(C,J)$ is $|C'| + |J|$, where $C'$ is $C$ without the bound constraints of the
variables $x_j$ with $j \in J$.

**Note that this definition of IIS follows the literature, except for the treatment of bounds of
integer variables. However, most current implementations differ in that they do not consider the
removal of integrality constraints.**

## Competition Task

Given infeasible MIP instances, produce for each an infeasible subsystem obtained by removing
constraints and/or integrality constraints.

The subsystem does not need to be an IIS; any infeasible subsystem is a valid output. Submissions
are evaluated on both the size of the subsystem and whether it is an IIS (see Evaluation Criteria).

### Instance Selection

The competition instances are linear MIPs with continuous and integer variables, supplied in MPS
format. RANGES, SOS, indicator and quadratic constraints, and semi-continuous variables are not
supported; ambiguous MPS files are rejected by the checker. Public
[sample instances and solutions](https://github.com/mixedinteger/MIPcc27/tree/main/smallset) are
available in the GitHub repository. They can be used as a starting point for investigating the
topic.

**Call for instances.** We welcome suggestions of interesting infeasible MIP instances from the
community, in particular real-world or structurally novel instances for which a small IIS would be
practically valuable or illustrative. Please send suggestions privately to
[the committee chair](mailto:bonami+mipcc27@gurobi.com).

### Input / Output

- **Input:** infeasible MIP instances in MPS format.
- **Output:** a JSON file specifying which elements to relax; elements not listed are kept. Optional
  feasibility certificates show that each single-element relaxation is feasible. See the published
  [solution format](https://github.com/mixedinteger/MIPcc27/blob/main/SOLUTION_FORMAT.md) and
  [program interface](https://github.com/mixedinteger/MIPcc27/blob/main/PROGRAM_INTERFACE.md).

## Evaluation Criteria

As is usual in the MIP competition, the jury will evaluate the submissions based both on performance
and innovation.

### Performance

The evaluation criteria will combine:

1. Correctness: the output is obtained only by constraint removal (including bound constraints)
   and/or integrality-constraint removal, and it is infeasible.
2. Size of the subsystem
3. Speed of computation
4. Whether the submitted subsystem is an IIS, supported by feasibility certificates (feasible
   solutions) for each single-element relaxation

The checker uses an absolute tolerance of $10^{-5}$ for certificates and for validating any
solver-produced feasible point used to check subsystem infeasibility. See the
[solution format](https://github.com/mixedinteger/MIPcc27/blob/main/SOLUTION_FORMAT.md#certificates)
and [solution checker](https://github.com/mixedinteger/MIPcc27#check-your-solutions) for the
verification rules.

### Innovation

The jury will also evaluate the quality and novelty of the approach described in the submitted
report.

## References

There is surprisingly little scientific literature on the topic of computing IISs for MIP. To our
knowledge, the only publications specifically on infeasibility analysis for MIP are the following
(see also the presentations listed at https://sites.google.com/view/johnchinneck/publications):

Guieu, O., & Chinneck, J. W. (1999). Analyzing infeasible mixed-integer and integer linear programs.
_INFORMS Journal on Computing_, **11**(1), 63–77. https://doi.org/10.1287/ijoc.11.1.63

Chinneck, J. W. (2008). _Feasibility and infeasibility in optimization: Algorithms and computational
methods_ (International Series in Operations Research & Management Science, Vol. 118). Springer.
https://doi.org/10.1007/978-0-387-74932-7
