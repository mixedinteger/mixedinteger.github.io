<img src="https://www.mixedinteger.org/2027/images/mip-2027.jpg" width="400">

# MIPcc27: The 2027 Land-Doig MIP Competition

> **Preliminary announcement.** The final rules, instance set, and submission instructions will be
> published in early October 2026.

## About

The computational development of optimization tools is a key component within the MIP community and
has proven to be a challenging task. It requires great knowledge of the well-established methods,
technical implementation abilities, as well as creativity to push the boundaries with novel
solutions.

In 2022, the annual [Mixed Integer Programming Workshop](https://www.mixedinteger.org/#mipworkshops)
established a computational competition in order to encourage and provide recognition to the
development of novel practical techniques within MIP technology. It was renamed in 2025 to honor
Ailsa H. Land and Alison G. Harcourt (née Doig), with permission, who proposed the first LP-based
branch-and-bound algorithm, a fundamental component of every modern MIP solver.

The competition is also supported by the
[Mixed Integer Programming Society (MIPS)](http://mixedinteger.org), a section of the
[Mathematical Optimization Society (MOS)](http://mathopt.org), via the
[MIP workshop](https://www.mixedinteger.org/2027/).

## Topic

The topic of the 2027 MIP competition is "Explainability and Small Infeasible Subsystems". The goal
is to advance the state of the art on computing small infeasible subsystems.

### Motivation

Explaining the outcomes of a MIP: for example, why a model is infeasible or how it could be made
feasible are natural questions that are difficult to answer due to the combinatorial nature of the
problem.

A widely used feature of commercial MIP solvers that aims at helping to explain infeasibility is the
so-called Irreducible Infeasible Subsystems (IIS). For a given MIP instance, an IIS is a subsystem
of linear constraints (including variable-bound constraints) that is still infeasible, but if any
single kept element is removed the resulting subsystem becomes feasible.

Producing a small IIS directly helps in having a short readable explanation of the infeasibility of
a potentially much larger model. Yet, while most modern commercial solvers have developed
(heuristic) methods to compute IIS in the context of MIP, there is very little scientific literature
(see references below) on good techniques to compute an IIS, and open-source implementations are
seldom (recently one was introduced in SCIP).

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

Note that this definition of IIS differs from the literature in that it allows for the removal of
integrality constraints.

## Competition Task

Given a collection of relatively simple infeasible MIP instances, produce for each instance an
infeasible subsystem obtained by removing constraints and/or integrality constraints.

The subsystem does not need to be an IIS; any infeasible subsystem is a valid output. Submissions are evaluated on both the size of the subsystem and whether it is an IIS (see Evaluation Criteria).

### Instance Selection

The public and hidden evaluation instance sets will be announced with the final rules in early October 2026.
Infeasible instances from [MIPLIB](https://miplib.zib.de/) can be used for starting.

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

## Submission Requirements

### Report

All participants must submit a written report of **10 pages maximum** plus references, in Springer
LNCS format. The report and the code must be submitted together.

The report must include the following information:

- A description of the method developed and implemented, including any necessary citations to the
  literature and software used.
- A section discussing the methodological and/or engineering innovations of the method (see
  Evaluation Criteria above). If you have any clever implementation techniques to showcase (e.g.
  performance optimization) please highlight (the jury will not check every details of the code).
- Computational results on the open competition test set, including a table of results produced with
  the benchmarking script.

### Code

The code should output a json file listing the constraint kept and optionally certificates that
removing them results in a feasible system.

## Rules and Eligibility

### Rules for Participation

- Participants must not be an organizer of the competition nor a family member of a competition
  organizer. Otherwise, there is no restriction on participation.
- Student participation is strongly encouraged.
- Should participants be related to organizers of the competition (e.g., students of an organizer),
  the rest of the committee will decide whether a conflict of interest is at hand and affected
  organizers should not be part of the jury for the final evaluation.
- Participants can be a single entrant or a team; there is no restriction on the size of teams.
- Questions may be directed to the competition committee by opening an
  [Issue](https://github.com/pobonomo/mip-competition-2027/issues) or
  [Discussion](https://github.com/pobonomo/mip-competition-2027/discussions) on the competition
  GitHub repository: [https://github.com/pobonomo/mip-competition-2027](https://github.com/pobonomo/mip-competition-2027),
  which will also host the checker and helper scripts.

### Technical Rules

- Competitors cannot use commercial solvers as subroutines of the final submission.

## To Be Announced (Early October 2026)

The following will be published with the final rules:

- Complete rules and problem/instance specification
- Public and hidden evaluation instance sets
- Instructions for submitting community instance suggestions
- Public release of the solution checker
- Registration form and process
- Code submission requirements (build, execution, interface, language)
- Organizing committee contact details
- Prizes and any associated support

## Timeline

| Date             | Milestone                                                                       |
| ---------------- | ------------------------------------------------------------------------------- |
| September 2026   | Competition topic announcement                                                  |
| Early October 2026 | Publication of rules, instance set and open registration                        |
| Mid January 2027 | Registration closes                                                             |
| Mid April 2027   | Final submission of solutions, evaluation on public and hidden set of instances |
| End of May 2027  | MIP Workshop: Winners announced                                                 |

## Organizing Committee

- [Pierre Bonami](https://pobonomo.github.io/) — Gurobi
- [Gerald Gamrath](https://www.zib.de/userpage/gamrath/) — Cardinal Operations
- [Shuvomoy Das Gupta](https://shuvomoy.github.io/) — Rice University
- [Reem Khir](https://sites.google.com/view/reemkhir/home) — Purdue University
- [Gioni Mexi](http://iol.zib.de/gionimexi) — Zuse Institute Berlin
- [Thiago Serra](https://thiagoserra.com) — University of Iowa

For questions, feedback, or inquiries, please visit the competition GitHub repository:
[https://github.com/pobonomo/MIPcc27](https://github.com/pobonomo/MIPcc27),
which hosts the solution checker and helper scripts. Participants can open an
[Issue](https://github.com/pobonomo/MIPcc27/issues) or start a
[Discussion](https://github.com/pobonomo/MIPcc27/discussions).

## References

There is surprisingly little scientific literature on the topic of computing IIS for MIP. Here are
three references that are relevant to the topic:

Guieu, O., & Chinneck, J. W. (1999). _Analyzing infeasible mixed-integer and integer linear
programs_. INFORMS Journal on Computing, **11**(1), 63–77. https://doi.org/10.1287/ijoc.11.1.63

Pfetsch, M. E. (2008). _Branch-and-cut for the maximum feasible subsystem problem_. SIAM Journal on
Optimization, **19**(1), 21–38. https://doi.org/10.1137/050645828

Amaldi, E., Pfetsch, M. E., & Trotter Jr., L. E. (2003). _On the maximum feasible subsystem problem,
IISs, and IIS-hypergraphs_. Mathematical Programming, **95**(3), 533–554.
