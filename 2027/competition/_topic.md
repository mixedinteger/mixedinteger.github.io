> **Preliminary announcement.** The final rules, instance set, and submission instructions still to
> be announced are listed on the [Rules page](rules.html#to-be-announced-early-october-2026); see
> also the [timeline](index.html#timeline).

## Topic

The topic of the 2027 MIP competition is "Explainability and Small Infeasible Subsystems". The goal
is to advance the state of the art on computing small infeasible subsystems.

### Motivation

Explaining the outcomes of a MIP: for example, why a model is infeasible or how it could be made
feasible are natural questions that are difficult to answer due to the combinatorial nature of the
problem.

A widely used feature of commercial MIP solvers for explaining infeasibility is the so-called
Irreducible Infeasible Subsystem (IIS). In the literature, an IIS is typically defined as a
subsystem of the original linear constraints of the MIP formulation (including variable-bound
constraints)that is infeasible, but becomes feasible if any single remaining element is removed.

Producing a small IIS directly helps provide a short, readable explanation of the infeasibility of a
potentially much larger model. Yet, while most modern commercial MIP solvers have developed
(heuristic) methods to compute IISs in the context of MIP, there is very little scientific
literature (see references below) on good techniques for computing IISs, and open-source
implementations are rare; one was recently introduced in SCIP.

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

Note that the system $Ax \le b$ may include variable-bound constraints, i.e., constraints of the
form $x_i \le u_i$ or $x_i \ge l_i$ for some $i \in \{1,\ldots,n\}$. All linear constraints are
indexed by set $M$.

In this competition, a subsystem is obtained from the original MIP by:

1. removing a subset of linear constraints (including bound constraints), and/or
2. removing integrality constraints for a subset of integer variables.

Let $C \subseteq M$ be the set of kept linear constraints and let $J \subseteq I$ be the set of
variables that remain integer. The resulting subsystem is denoted by $MIP(C,J)$ and is formally
defined as

$$
\begin{aligned}
\min \quad & c^T x\\
\mathrm{s.t.} \quad & a_r^T x \le b_r && \forall r \in C, \\
& x_j \in \mathbb{Z} && \forall j \in J, \\
& x \in \mathbb{R}^n.
\end{aligned}
$$

For an infeasible instance of MIP, we say that $MIP(C,J)$ is an IIS if:

1. $MIP(C,J)$ is infeasible;
2. for every constraint $r \in C$, the subsystem $MIP(C \setminus \{r\}, J)$ is feasible;
3. for every integer variable $j \in J$, the subsystem $MIP(C, J \setminus \{j\})$ is feasible.

This is a single-element irreducibility definition. In particular, relaxing a variable bound is
treated as removing the corresponding bound constraint.

**Note that this definition of IIS differs from the literature in that it allows for the removal of
integrality constraints.**

## Competition Task

Given a collection of relatively simple infeasible MIP instances, produce for each instance an
infeasible subsystem obtained by removing constraints and/or integrality constraints.

The subsystem does not need to be an IIS; any infeasible subsystem is a valid output. Submissions
are evaluated on both the size of the subsystem and whether it is an IIS (see Evaluation Criteria).

### Instance Selection

The public and hidden evaluation instance sets will be announced with the final rules in early
October 2026. Infeasible instances from [MIPLIB](https://miplib.zib.de/) can be used as a starting
point for investigating the topic.

**Call for instances.** We welcome suggestions of interesting infeasible MIP instances from the
community, in particular real-world or structurally novel instances for which a small IIS would be
practically valuable or illustrative. Instructions for submitting instance suggestions will be
published with the final rules.

### Input / Output

- **Input:** infeasible MIP instances in MPS format.
- **Output:** list of constraints and integrality constraints that are kept in the infeasible
  subsystem.

## Evaluation Criteria

As is usual in the MIP competition, the jury will evaluate the submissions based both on performance
and innovation.

### Performance

The evaluation criteria will combine:

1. Correctness: the output is obtained only by constraint removal (including bound constraints)
   and/or integrality-constraint removal, and it is infeasible.
2. Size of the subsystem
3. Speed of computation
4. Whether the submitted subsystem is an IIS, supported by feasible certificates for each
   single-element relaxation

### Innovation

The jury will also evaluate the quality and novelty of the approach described in the submitted
report.

## References

There is surprisingly little scientific literature on the topic of computing IIS for MIP. Here are
three references that are relevant to the topic:

Guieu, O., & Chinneck, J. W. (1999). _Analyzing infeasible mixed-integer and integer linear
programs_. INFORMS Journal on Computing, **11**(1), 63–77. https://doi.org/10.1287/ijoc.11.1.63

Pfetsch, M. E. (2008). _Branch-and-cut for the maximum feasible subsystem problem_. SIAM Journal on
Optimization, **19**(1), 21–38. https://doi.org/10.1137/050645828

Amaldi, E., Pfetsch, M. E., & Trotter Jr., L. E. (2003). _On the maximum feasible subsystem problem,
IISs, and IIS-hypergraphs_. Mathematical Programming, **95**(3), 533–554.
