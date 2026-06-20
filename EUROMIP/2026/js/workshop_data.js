export const workshop_data = [
  {
    improper: "",
    firstname: "",
    lastname: "",
    affiliation: "",
    website: "",
    title: "",
    time: "",
    date: "",
    sortdate: "",
    youtube: "",
    picture: "",
    abstracttext: ""
  },
  {
    firstname: "Stefan",
    lastname: "Kober",
    affiliation: 'Université Libre de Bruxelles',
    website: 'https://sites.google.com/view/stefankober/',
    title: 'Recent advances on integer programming with bounded subdeterminants',
    abstracttext: 'It is a notorious open question whether integer programs (IPs), with an integer constraint matrix M whose subdeterminants are bounded by a constant in absolute value, can be solved in polynomial time. In recent years, different versions of this question have been studied from perspectives including geometry, IP theory, graph and matroid theory and more. We give an overview on recent progress towards this question and the rich combinatorial structures hidden within.',
    picture: 'images/speakers/stefan_kober.jpg'
  },
  {
    firstname: "Ayse Nur",
    lastname: "Arslan",
    affiliation: 'Universite de Bordeaux',
    website: 'https://aysnrarsln.github.io/',
    picture: 'images/speakers/ayse_nur_arslan.jpeg',
    title: 'Decomposition algorithms for two-stage robust optimization problems',
    abstracttext: 'In this talk we consider two-stage robust optimization problems and propose decomposition algorithms for their solution. The first class of problems we consider concern binary recourse with only objective function uncertainty. For these problems we propose a convexification approach based on Dantzig-Wolfe decomposition and show how the problem can be solved to exact optimality using the column-generation algorithm. We provide numerical results on the comparison between this exact approach and the approximate K-adaptability algorithm. We also give insights on how the same convexification approach can be valuable in other contexts such as two-stage stochastic programming. The second class of problems concern problems with continuous and fixed recourse. These problems have been the subject of exact solution approaches, notably, constraint generation (CG) and constraint-and-column generation (CCG). We present an approach reposing on a novel reformulation of the problem with an exponential number of semi-infinite constraints and develop a nested decomposition algorithm to deal with the exponential and semi-infinite natures of our formulation separately. We provide numerical results that showcase the superior performance of our proposed approach compared to the state-of-the-art and evaluate the contribution of different algorithmic components.'
  },
  {
    firstname: "Sophie",
    lastname: "Huiberts",
    affiliation: 'LIMOS, Clermont Auvergne University',
    website: 'https://sophie.huiberts.me/',
    title: 'Analyzing the simplex method by-the-book, or, what theory can learn from practice',
    abstracttext: 'The simplex method is an algorithm for linear programming, and this algorithm is much faster than theory is able to explain. In this talk I will describe a new theoretical framework we introduced to address this question. Under this framework, we prove new strong running time guarantees, using mathematical assumptions taken from software user manuals. I will discuss which features of real-world software and LPs we have managed to theoretically capture for this purpose, and what will come next.',
    picture: 'images/speakers/sophie_huiberts.jpg'
  },
  {
    firstname: "Emiliano",
    lastname: "Traversi",
    affiliation: 'ESSEC',
    website: 'https://faculty.essec.edu/en/cv/traversi-emiliano/',
    title: 'TBA'
  },
  {
    firstname: "Leo",
    lastname: "Liberti",
    affiliation: 'LIX CNRS, Ecole Polytechnique',
    website: 'https://www.lix.polytechnique.fr/~liberti/',
    title: 'Random projections in mathematical programming: recent advances',
    abstracttext: 'In this talk I will briefly survey previous work about the application of random projections to mathematical programming, and then talk about recent advances, specifically about quadratically constrained quadratic programs, as well as on a specific MINLP problem, i.e.~the minimum sum-of-squares clustering.\n\nCo-authors: Benedetto Manca and Pierre-Louis Poirion',
    picture: 'images/speakers/leo_liberti.png'
  },
  {
    firstname: "Elisabeth",
    lastname: "Gaar",
    affiliation: 'Universität Augsburg',
    website: 'https://www.uni-augsburg.de/de/fakultaet/mntf/math/prof/opt/team/gaar/',
    title: 'MIP approaches for the $p$-$\alpha$-closest-center problem',
    abstracttext: 'We study a recently emerged resilient variant of the well-known facility location $p$-center problem, the $p$-$\alpha$-closest-center problem. In this problem, we are given a set of customer demand points, a set of possible facility locations, distances between each customer demand point and each possible facility location, as well as two integers $p$ and $\alpha$. The goal of the $p$-$\alpha$-closest-center-problem is to open a facility at $p$ of the possible facility locations such that the maximum sum of the distances from any customer to its $\alpha$ closest open facilities is minimized.\nIn this talk, we several novel mixed-integer programming (MIP) formulations of the $p$-$\alpha$-closest center problem and strengthen them by adding valid inequalities. We also conduct a polyhedral study, present iterative lifting procedures that exploit combinatorial information to derive tighter linear programming relaxations and characterize the best lower bounds obtainable with it. We also carry out a computational study, applying our improved relaxations within a branch-and-cut framework, which we enhance with a starting and a primal heuristic, variable fixings and separating inequalities. This is joint work with Sara Joosten and Markus Sinnl.',
    picture: 'images/speakers/elisabeth_gaar.jpg'
  },
  {
    firstname: "Martin",
    lastname: "Schmidt",
    affiliation: 'Trier University',
    website: 'https://martinschmidt.squarespace.com/about',
    title: 'Branch-and-Cut for Mixed-Integer Nash Equilibrium Problems',
    abstracttext: "We consider Nash equilibrium problems with mixed-integer variables in which each player solves a mixed-integer optimization problem parameterized in the rivals' strategies. We distinguish between standard Nash equilibrium problems (NEP), where the parameterization acts only on the players' cost functions and generalized Nash equilibrium problems (GNEPs), where, additionally, the strategy spaces of the players may depend on the rivals' strategies. We introduce a branch-and-cut (B&C) algorithm for such mixed-integer games that, upon termination, either computes a pure Nash equilibrium or decides their non-existence. The main idea is to reformulate the equilibrium problem as a suitable bilevel problem based on the Nikaido-Isoda function of the game. We then use bilevel-optimization techniques to get a computationally tractable relaxation of this reformulation and embed it into a B&C framework. We derive sufficient conditions for the existence of suitable cuts and finite termination of our method depending on the setting. For GNEPs, we adapt the idea of intersection cuts from bilevel optimization and mixed-integer linear optimization. We can guarantee the existence of such cuts under suitable assumptions, which are particularly fulfilled for pure-integer GNEPs with decoupled concave objectives and linear coupling constraints. For NEPs, we show that suitable cuts always exist via best-response inequalities and prove that our B&C method terminates in finite time whenever the set of best-response sets is finite. We show that this condition is fulfilled for the important special cases of (i) players' cost functions being concave in their own continuous strategies and (ii) the players' cost functions only depending on their own strategy and the rivals' integer strategy components. Finally, we present preliminary numerical results for two different types of knapsack games, a game based on capacitated flow problems, and integer NEPs with quadratic objectives.",
    picture: 'images/speakers/Portrait-Martin-Schmidt19930_pp.webp'
  },
  {
    firstname: "Antonio Maria",
    lastname: "Sudoso",
    affiliation: 'Sapienza University',
    website: 'https://sites.google.com/view/antoniosudoso',
    title: 'Exact Methods for Variance Minimization Problems in Data Analysis',
    abstracttext: 'Variance minimization plays a fundamental role in subset selection, clustering and high-dimensional data analysis, where the objective is to identify compact, homogeneous, and meaningful structures within complex datasets. Despite their broad applicability, these problems typically lead to challenging large-scale combinatorial optimization models whose exact solution remains computationally demanding. In this talk, we discuss recent advances in the design of exact algorithms for variance-based optimization problems, with particular emphasis on strong relaxations with theoretical guarantees and decomposition techniques.',
    picture: 'images/speakers/antonio_sudoso.jpeg'
  },
  {
    firstname: "Yasmine",
    lastname: "Beck",
    affiliation: 'Eindhoven University of Technology',
    website: 'https://yasminebeck.github.io/',
    title: 'Exact Methods for Recoverable Robust Combinatorial Optimization Problems under Budgeted Uncertainty',
    abstracttext: 'We study recoverable robust combinatorial optimization problems in which a decision maker solves an optimization problem subject to objective uncertainty. The model follows a two-stage robust setup in which the decision maker first commits to an initial solution and may adjust it after the uncertainty is revealed. In this talk, we focus on the setting in which these adjustments correspond to revoking some of the initial commitments. The underlying uncertainty is modeled using a budgeted uncertainty set so that the decision maker only hedges against a limited number of deviations in the uncertain parameters. We present different equivalent reformulations of the recoverable robust problem, which can be tackled using (i) general-purpose MILP solvers, (ii) branch-and-cut methods, or (iii) column-and-constraint generation algorithms. The performance of all proposed methods is assessed and compared in a computational study.',
    picture: "images/speakers/yasmine_beck.jpeg"
  },
  {
    firstname: "Jannis",
    lastname: "Kurtz",
    affiliation: 'University of Amsterdam',
    website: 'https://www.janniskurtz.eu/',
    title: 'K-Adaptability in Two-Stage Integer Robust Optimization',
    abstracttext: 'In the realm of robust optimization the k-adaptability approach is one promising method to derive near-optimal solutions for two-stage robust optimization problems with integer decision variables. Instead of allowing all possible second-stage decisions, the k-adaptability approach aims at calculating a limited set of k such decisions already in the first-stage before the uncertainty reveals. The parameter k can be adjusted to control the quality of the approximation. However, until recently, not much was known on how many solutions k are needed to achieve an optimal or approximate solution for the two-stage robust problem. In this talk we approach this question by deriving bounds on k for objective and constraint uncertainty which lead to optimal solutions or solutions with a given approximation guarantee. The results give new insights on how many solutions are needed for problems as the decision dependent information discovery problem or the capital budgeting problem with constraint uncertainty.',
    picture: 'images/speakers/jannis_kurtz.jpg'
  },
  {
    firstname: "Karen",
    lastname: "Aardal",
    affiliation: 'Delft University of Technology',
    website: 'https://diamhomes.ewi.tudelft.nl/~kaardal/',
    title: 'TBA'
  },
  {
    firstname: "Bissan",
    lastname: "Ghaddar",
    affiliation: 'Ivey Business School and IE University',
    website: 'https://www.ie.edu/university/about/faculty/bissan-ghaddar/',
    title: 'Machine Learning-Enhanced Non-Linear Optimization',
    abstracttext: 'Nonlinear optimization problems present significant computational challenges due to their inherent nonconvexity. In this talk, we explore how machine learning can be integrated into key algorithmic components of nonlinear optimization solvers to enhance their efficiency. We examine how learning can be used to predict branching decisions and variable selection within branch-and-bound frameworks, and to generate effective cutting planes or conic constraints that strengthen relaxations of these problems. We then extend this work to learning-guided decomposition for semidefinite programming relaxations, particularly for large-scale nonlinear optimization problems arising in energy systems. We highlight how structural properties such as sparsity and network topology can be leveraged to inform learning models that guide the decomposition of semidefinite relaxations.'
  },
  {
    firstname: "Vanesa",
    lastname: "Guerrero",
    affiliation: 'Universidad Carlos III de Madrid',
    website: 'https://researchportal.uc3m.es/display/inv45738',
    title: 'Mixed-integer programming for model selection in smooth and networked data',
    abstracttext: 'Mixed-integer programming provides a powerful framework for model selection in complex and structured data settings. In this talk, we address two such scenarios: smooth regression models with shape constraints and learning tasks over distributed networks, where interpretability and sparsity are key challenges. On the one hand, for shape-constrained smooth additive models, we study variable selection through a best subset approach, leading to formulations that incorporate both structural constraints, modeled through a conic optimization framework, and sparsity, introduced via binary decision variables. To tackle the resulting problem, we develop tight continuous relaxations based on perspective formulations. On the other hand, we consider multi-task learning in networked data, where observations are distributed across nodes. We propose methods to partition the network into connected clusters, assigning each cluster a shared predictive model. This is achieved either by selecting existing local models or by learning new ones from pooled data, while enforcing connectivity constraints. We develop a solution approach based on the Variable Neighborhood Search metaheuristic and, for specific network topologies, an exact dynamic programming approach. Experiments on synthetic and real datasets demonstrate that the proposed methods achieve competitive performance while producing interpretable and parsimonious models. Overall, these works highlight the potential of optimization-based approaches for model selection in nonlinear and network-structured data.',
    picture: 'images/speakers/vanesa_guerro.jpg'
  },
  {
    firstname: "Fritz",
    lastname: "Eisenbrand",
    affiliation: 'EPFL',
    website: 'https://people.epfl.ch/friedrich.eisenbrand',
    title: 'A parameterized linear formulation of the integer hull',
    abstracttext: "Let A ∈ ℤ<sup>m × n</sup> be an integer matrix with components bounded by Δ in absolute value. Cook et al. (1986) have shown that there exists a universal matrix B ∈ ℤ<sup>m' × n</sup> with the following property: For each b ∈ ℤ<sup>m</sup>, there exists t ∈ ℤ<sup>m'</sup> such that the integer hull of the polyhedron P = { x ∈ ℝ<sup>n</sup> : Ax ≤ b } is described by P<sub>I</sub> = { x ∈ ℝ<sup>n</sup> : Bx ≤ t }. Our main result is that t is an affine function of b as long as b is from a fixed equivalence class of the lattice D · ℤ<sup>m</sup>. Here D ∈ ℕ is a number that depends on n and Δ only. Furthermore, D as well as the matrix B can be computed in time depending on Δ and n only. An application of this result is the solution of an open problem posed by Cslovjecsek et al. (SODA 2024) concerning the complexity of 2-stage-stochastic integer programming problems.\n\nThe main tool of our proof is the classical theory of Chv&aacute;tal-Gomory cutting planes and the elementary closure of rational polyhedra.\n\nThis is joint work with Thomas Rothvoss",
    // abstracttext: "Let $A \in \mathbb{Z}^{m \times n}$ be an integer matrix with components bounded by $\Delta$ in absolute value. Cook et al. (1986) have shown that there exists a universal matrix $B \in \mathbb{Z}^{m' \times n}$ with the following property: For each $b \in \mathbb{Z}^m$, there exists $t \in \mathbb{Z}^{m'}$ such that the integer hull of the polyhedron $P = \{ x \in \mathbb{R}^n : Ax \leq b\}$ is described by $P_I = \{ x \in \mathbb{R}^n : Bx \leq t\}$. Our main result is that $t$ is an affine function of $b$ as long as $b$ is from a fixed equivalence class of the lattice $D \cdot \mathbb{Z}^m$. Here $D \in \mathbb{N}$ is a number that depends on $n$ and $\Delta$ only. Furthermore, $D$ as well as the matrix $B$ can be computed in time depending on $\Delta$ and $n$ only. An application of this result is the solution of an open problem posed by Cslovjecsek et al. (SODA 2024) concerning the complexity of 2-stage-stochastic integer programming problems.\n\nThe main tool of our proof is the classical theory of Chv&aacute;tal-Gomory cutting planes and the elementary closure of rational polyhedra.\n\nThis is joint work with Thomas Rothvoss"
    picture: 'images/speakers/fritz_eisenbrand.jpg'
  },
  {
    firstname: "Kübra",
    lastname: "Taninmis",
    affiliation: 'Koc Ünicersitesi',
    website: 'https://gsse.ku.edu.tr/en/programs/industrial-engineering-and-operations-management/faculty/?detail=true&id=ktaninmis',
    title: 'Fair Influence Maximization: an Exact Approach',
    abstracttext: 'Influence Maximization Problem aims at identifying a small set of seed nodes to maximize the expected spread of information on a social network. When the community structure inherent in a social network is overlooked, the optimal seed set often results in a highly imbalanced spread across different groups. This imbalance is undesirable in scenarios involving socially beneficial information, such as job postings and public health outreach campaigns. In contrast to the heuristic approaches commonly discussed in the literature, our objective is to determine the optimal seed set that meets a specified fairness criterion and to assess the price-of-fairness in a more reliable way. To achieve this, we introduce a mixed-integer programming model into which established fairness principles can be directly incorporated. We develop a unified Benders decomposition framework for several fairness principles to address the scalability challenges. Preliminary results demonstrate that the proposed decomposition method significantly outperforms standard commercial solvers.',
    picture: 'images/speakers/kubra_taninmis.png'
  },
  {
    firstname: "Ruth",
    lastname: "Misener",
    affiliation: 'Imperial College London',
    website: 'https://profiles.imperial.ac.uk/r.misener',
    title: 'Optimizing over graphs: Challenges, Formulations, and Applications',
    abstracttext: 'Applications involving optimization over graphs include molecular design, graph neural network verification, neural architecture search, etc. This talk discusses formulating graph spaces using mixed-integer optimization and incorporating application-specific constraints. We discuss computational challenges with these mixed-integer optimization formulations and zoom in on the practical implications for these applications. We mention what has been done (by both ourselves and others) and what other research still needs to be done.\n\nCo-authors: Shiqiang Zhang, Yilin Xie, Christopher Hojny, Juan Campos, Jixiang Qing, Christian Feldmann, David Walz, Frederik Sandfort, Miriam Mathea, Calvin Tsay',
    picture: 'images/speakers/ruth_misener.jpg'
  },
  {
    firstname: "Andrea",
    lastname: "Lodi",
    affiliation: 'Cornell Tech',
    website: 'https://tech.cornell.edu/people/andrea-lodi/',
    title: 'TBA',
    picture: 'images/speakers/andrea_lodi.jpg'
  },
  {
    firstname: "Péter",
    lastname: "Biró",
    affiliation: 'Hungarian Academy of Sciences',
    website: 'https://mechanismdesign.eu/biro/',
    title: 'Smart Lotteries in School Choice: Ex-ante Pareto-Improvement with Ex-post Stability',
    abstracttext: "In a typical school choice application, the students have strict preferences over the schools while the schools have coarse priorities over the students based on their distance and their enrolled siblings. The outcome of a centralized admission mechanism is then usually obtained by the Deferred Acceptance (DA) algorithm with random tie-breaking. Therefore, every possible outcome of this mechanism is a stable solution for the coarse priorities that will arise with certain probability. This implies a probabilistic assignment, where the admission probability for each student-school pair is specified. In this paper, we propose a new efficiency-improving stable `smart lottery' mechanism. We aim to improve the probabilistic assignment ex-ante in a stochastic dominance sense, while ensuring that the improved random matching is still ex-post stable, meaning that it can be decomposed into stable matchings regarding the original coarse priorities. Therefore, this smart lottery mechanism can provide a clear Pareto-improvement in expectation for any cardinal utilities compared to the standard DA with lottery solution, without sacrificing the stability of the final outcome. We show that although the underlying computational problem is NP-hard, we can solve the problem by using advanced optimization techniques such as integer programming with column generation. We conduct computational experiments on generated and real instances. Our results show that the welfare gains by our mechanism are substantially larger than the expected gains by standard methods that realize efficiency improvements after ties have already been broken.",
    picture: 'images/speakers/biro-peter.jpg'
  },
  {
    firstname: "Marco",
    lastname: "Lübbecke",
    affiliation: 'RWTH Aachen',
    website: 'https://www.or.rwth-aachen.de/en/details-staff/luebbecke.html',
    title: 'TBA'
  },
  {
    firstname: "Petra",
    lastname: "Mutzel",
    affiliation: 'Universität Bonn',
    website: 'https://ca.cs.uni-bonn.de/doku.php?id=people:mutzel',
    title: 'Graph Edit Distance: Theory, Models, and Algorithms',
    abstracttext: 'The Graph Edit Distance (GED) is a fundamental measure of graph similarity with applications in  graph learning, pattern recognition, and network analysis. Its computation is NP-hard and gives rise to challenging combinatorial optimization problems.\n\nThis talk presents recent advances in the theory and computation of GED. After a brief introduction to graph similarity and edit-distance-based approaches, I will focus on FORI, a new integer linear programming formulation for the exact computation of GED. Particular emphasis is placed on the modeling aspects of the problem and on a theoretical comparison of competing formulations. We analyze the strength of their linear programming relaxations and establish dominance relations that explain the observed computational performance. The theoretical findings are complemented by algorithmic developments and computational results showing significant improvements over previous exact approaches.\n\nThis talk is based on joint work with Andrea D’Ascenzo (Gran Sasso Science Institute L’Aquila), Julian Meffert (University of Bonn), and Fabrizio Rossi (University of L’Aquila).',
    picture: 'images/speakers/petra_mutzel.jpeg'
  },
  {
    firstname: "Jon",
    lastname: "Lee",
    affiliation: 'University of Michigan',
    website: 'https://sites.google.com/site/jonleewebpage/',
    title: 'Extended-Variable Relaxations for the Constrained Generalized Maximum-Entropy Sampling Problem',
    abstracttext: 'The generalized maximum-entropy sampling problem (GMESP) is to select an order-s principal submatrix from an order-n covariance matrix, to maximize the product of its t greatest eigenvalues, 0 < t <= s < n. Introduced more than 25 years ago, GMESP is a natural generalization of two fundamental problems in statistical design theory: (i) maximum-entropy sampling problem (MESP); (ii) binary D-optimality (D-Opt). In the general case, it can be motivated by a selection problem in the context of principal component analysis (PCA). We approach GMESP as a challenging nonlinear integer optimization problem, and we aim at developing effective B&amp;B approaches. We present (i) non-convex extended-variable formulations, (ii) first non-convex and then convex continuous relaxations, (iii) results analyzing our new (upper) bounds and demonstrating some relations between different bounds, including bounds from the literature and our new bounds, (iv) theory related to how to carry out B&B (in particular, variable fixing and subproblem construction), and (v) favorable numerical results. This is joint work with Kurt Anstreicher, Marcia Fampa, and Gabriel Ponte.',
    picture: 'images/speakers/jon_lee.jpg'
  },
];

