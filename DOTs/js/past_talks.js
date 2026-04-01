export const past_speakers = [
  {
    name: "Claudia D'Ambrosio",
    affiliation: 'CNRS & École Polytechnique',
    website: 'https://www.lix.polytechnique.fr/~dambrosio/',
    title: 'Perspective Formulations for Piecewise Convex Functions',
    date: 'March 27, 2026',
    youtube: 'https://www.youtube.com/watch?v=RDDBdmKzSw4',
    picture: 'img/claudia_dambrosio.png',
    abstracttext: "In this talk, we focus on mixed integer nonlinear problems with univariate piecewise convex inequalities. We generalize the classic formulation for piecewise linear functions to the case of piecewise convex ones and strengthen them through perspective reformulation. Moreover, we compare the different formulations and we show theoretically that they are not equivalent, contrarily to what has been shown in the literature for the piecewise linear formulations. Computational results on two classes of problems are presented, confirming the theoretical findings."
  },
  {
    name: 'Oktay Gunluk',
    affiliation: 'Georgia Tech',
    website: 'https://sites.google.com/site/oktaygunlukresearch/Home',
    title: 'Recovering Dantzig-Wolfe Bounds by Cutting Planes',
    date: 'March 27, 2026',
    youtube: 'https://www.youtube.com/watch?v=OaE0_-rR2HM',
    picture: 'img/oktay_gunluk.png',
    abstracttext: "Dantzig-Wolfe (DW) decomposition is a well-known technique in mixed-integer programming (MIP) for decomposing and convexifying constraints to obtain potentially strong dual bounds. We investigate cutting planes that can be derived using the DW decomposition algorithm and show that these cuts can provide the same dual bounds as DW decomposition. More precisely, we generate one cut for each DW block, and when combined with the constraints in the original formulation, these cuts imply the objective function cut one can simply write using the DW bound. This approach typically leads to a formulation with lower dual degeneracy that consequently has a better computational performance when solved by standard MIP solvers in the original space. We also discuss how to strengthen these cuts to improve the computational performance further.\n\nJoint work with Rui Chen, and Andrea Lodi"
  },
  {
    name: 'Vera Traub',
    affiliation: 'ETH Zurich',
    website: 'https://people.inf.ethz.ch/vtraub/',
    title: 'Unsplittable Cost Flows from Unweighted Error-Bounded Variants',
    date: 'February 27, 2026',
    youtube: 'https://www.youtube.com/watch?v=bQWyysxmAag',
    picture: 'img/vera_traub.png',
    abstracttext: "A famous conjecture of Goemans on single-source unsplittable flows states that one can turn any fractional flow into an unsplittable one of no higher cost, while increasing the load on any arc by at most the maximum demand. Despite extensive work on the topic, only limited progress has been made. Recently, Morell and Skutella suggested an alternative conjecture, stating that one can turn any fractional flow into an unsplittable one without changing the load on any arc by more than the maximum demand.\n\nWe show that their conjecture implies Goemans' conjecture (with a violation of twice the maximum demand). To this end, we generalize a technique of Linhares and Swamy, used to obtain a low-cost chain-constrained spanning tree from an algorithm without cost guarantees. Whereas Linhares and Swamy's proof relies on Langrangian duality, we provide a very simple elementary proof of a generalized version, which we hope to be of independent interest.\n\nThis is joint work with Chaitanya Swamy, Laura Vargas Koch, Rico Zenklusen"
  },
  {
    name: 'Ozlem Ergun',
    affiliation: 'Northwestern University',
    website: 'https://coe.northeastern.edu/people/ergun-ozlem/',
    title: 'Enhancing the Effectiveness and Responsiveness of Humanitarian Food Aid Delivery Services Through Data-Driven Optimization',
    date: 'February 27, 2026',
    youtube: 'https://www.youtube.com/watch?v=Vk8DDLrUpoI',
    picture: 'img/ozlem_ergun.png',
    abstracttext: "Humanitarian organizations must plan and operate two distinct aid delivery services on a shared network for responding to both ongoing (OG) demand in a reliable and cost-efficient way and sudden-onset (SO) emergencies rapidly. This paper studies the joint planning problem and presents scenario-informed, data-driven delivery service recommendations that preserve managerial interpretability. We develop an integrated two-stage model to jointly plan OG and SO services over a shared network under common operational constraints. In the first stage, the model optimizes over which warehouses to open and SO-dedicated safety stock levels; in the second stage, operations over the network are decided for serving the known OG demand and the SO requests revealed over time. To support planning under limited historical data, a deep-learning pipeline labels historical demand as OG or SO and generates realistic, diverse demand scenarios that preserve temporal and spatial structure while introducing controlled variation. Rather than solving a single large stochastic program, we solve the model per scenario to build a reusable decision library and reveal how first-stage designs vary with scenario conditions. From this library, we derive (i) a baseline recommendation using only historical scenarios and (ii) a pattern-based recommendation based on the frequency of first-stage decision patterns over all scenarios generated. Using a pre-specified demand-feature set, we train a feature-to-decision recommender that maps scenario features to representative first-stage designs, enabling fast, interpretable, scenario-specific guidance without re-solving the optimization model. Computational experiments show that, relative to the historical baseline, pattern-based recommendation improves per unit cost efficiency and increases early fulfillment and service coverage. Feature importance analysis from feature-to-decision recommender indicates that total sudden-onset demand and the budget constraints are the primary decision drivers, with commodity compositions, seasonality, and geography refining the recommended design."
  },
  {
    name: 'Emma Johnson',
    affiliation: 'Sandia National Laboratories',
    website: '',
    title: 'Defensive Flow Interdiction: How to Attack Your Own Network',
    date: 'January 30, 2026',
    youtube: '',
    picture: 'img/emma_johnson.jpg',
    abstracttext: "Network interdiction involved expending a limited budget destroying infrastructure in a network to maximally reduce some functionality of the network. Typically, the network is owned by an adversary, and we are trying to interdict their use of the network. We introduce a novel variant of the maximum flow interdiction problem in which the adversary is using a network we own. We wish to reclaim our network and still have it as functional as possible for our goals. To model this situation we assume we have a set of  'bad pairs', which are pairs of nodes the adversary has taken over. The adversary needs flows above a minimum threshold between bad pairs to achieve its goal. We are also given a set of 'good pairs', which represent the functionality of our own network we wish to preserve after the interdiction. We are required to maintain flow above a minimum threshold between good pairs. Our objective it so maximize the number of bad pairs whose residual flow capacity drops below their functional threshold. We formulate this problem as a mixed-integer linear program (MILP) and also introduce a fast and effective MILP-based primal heuristic motivated by a previous pseudo-approximation for classic max-flow interdiction. We show the accuracy and computational benefits of the heuristic relative to the MILP formulation on a computational study on graph instances from the SNAP and DIMACS repositories."
  },
  {
    name: 'Yongjia Song',
    affiliation: 'Clemson University',
    website: 'https://sites.google.com/site/yongjiasongshom',
    title: 'A Robust Bilevel Network Interdiction Problem',
    date: 'January 30, 2026',
    youtube: 'https://www.youtube.com/watch?v=3-kasFEpXvo',
    picture: 'img/yongjia_song.png',
    abstracttext: "In this talk, we investigate a robust bilevel network interdiction problem motivated by applications in human trafficking disruption. In this problem, the follower, who we assume to be rational, will solve a minimum cost network flow problem on a network whose arcs may be interdicted/removed by the leader. The leader, who interdicts the network, optimizes their own objective computed using the optimal flow obtained in the follower’s problem. We consider the case where the leader does not know the follower’s cost vector, but only that it belongs to a given uncertainty set, while the follower has complete knowledge about their own parameters. We present a column-and-constraint generation (C&CG) method to solve the optimistic version of this problem. We discuss the difficulties in solving the standard subproblem from the C&CG method and propose a method to solve the subproblem by exploiting the structure of the uncertainty set. We demonstrate the effectiveness of the proposed approach through computational experiments with synthetic human trafficking networks."
  },
];

