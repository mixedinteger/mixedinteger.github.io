# MIPcc25: The MIP Workshop 2025 Computational Competition


## About

The computational development of optimization tools is a key component within the MIP community and has proven to be a challenging task. It requires great knowledge of the well-established methods, technical implementation abilities, as well as creativity to push the boundaries with novel solutions. In 2022, the annual [Mixed Integer Programming Workshop](https://www.mixedinteger.org/) established a computational competition in order to encourage and provide recognition to the development of novel practical techniques within MIP technology. This year, the competition will focus on primal heuristics for mixed-integer quadratic and quadratically-constrained problems.


## The Challenge: MIP Quadratic Primal Heuristics

Here we consider mixed-integer problems with some quadratic functions (objective and/or constraints), technically both MIQP and MIQCQP. A so-called primal heuristic is an algorithm intended to quickly find good feasible solutions.

The ability to quickly find feasible solutions is an essential component of state-of-the-art MIP solvers. According to the computational survey [A computational study of primal heuristics inside an MI(NL)P solver](https://link.springer.com/article/10.1007/s10898-017-0600-3), disabling the primal heuristics significantly reduces the overall performance of the SCIP solver. Obtaining a good feasible solution early in the solution process can, for example, reduce the size of the branch-and-bound tree, and for many end users a good feasible solution can be more important than a tight dual bound. 

Several primal heuristics have been developed for MILP and for more general MINLP. But, there has been less focus on MIQP/MIQCQP. The aim of the competition is to fill this gap.

A good primal heuristic algorithm should be able to find a feasible solution quickly, and possibly improve it. Therefore, we ask participants to develop code that returns several solutions across time, not just the best found solution. More instructions are given in the technical details.

The task is to provide:

*   a **primal heuristic code** implementing one (or several) primal heuristics. The code must write a solution file for each solution found and a result file as described in the *Technical Rules* section below.
*   a **written report** describing the methodology and results.

Finalists will be provided travel support to present their methods at the [MIP Workshop 2025](https://www.mixedinteger.org/2025/) held in June 2025. High-quality submissions will be invited to an expedited review process in [Mathematical Programming Computation](https://www.springer.com/journal/12532).


## Timeline



*   December 4, 2024: Publication of the topic, rules and set of test problems.
*   **January 31, 2025**: Registration deadline for participation
*   **March 16, 2025**: Submission deadline for report and code (**Anytime on Earth**)
*   Early April 2025: Notification of results
*   June 2025: Presentations of the finalists at the MIP Workshop


## Awards



*   The jury will select up to three finalists to present their work at the MIP Workshop 2025. The final winner will be announced at the MIP Workshop 2025.
*   One representative of each finalist will receive travel support to MIP and free registration.
*   The performance of non-finalist submissions will not be published.
*   High-quality submissions will receive an expedited review process in [Mathematical Programming Computation](https://www.springer.com/journal/12532).
*   The jury may present an award to recognize outstanding student submissions. For these submissions, senior supervisors may be part of the team, but the implementation work must have been fully conducted by students. Students should not have received their first PhD degree on March 1st, 2025.


## Organizing committee



*   [Timo Berthold](https://www.zib.de/userpage/berthold/), FICO
*   [Chen Chen](https://u.osu.edu/chen/), Ohio State University
*   [Aleksandr Kazachkov](https://akazachk.github.io/), University of Florida
*   [Jan Kronqvist](https://www.kth.se/profile/jankr), KTH Royal Institute of Technology, Chair
*   [Dimitri Papageorgiou](https://www.linkedin.com/in/dimitri-papageorgiou-710b158/), ExxonMobil
*   [Domenico Salvagnin](https://www.dei.unipd.it/~salvagni/), DEI, University of Padova
*   [Christian Tjandraatmadja](https://research.google/people/christiantjandraatmadja/?&type=google), Google Research
*   [Calvin Tsay](https://www.doc.ic.ac.uk/~ctsay/), Imperial College London


## Rules and Evaluation Procedure


### Rules for Participation



*   Participants must not be an organizer of the competition nor a family member of a competition organizer.
*   Otherwise, there is no restriction on participation.
*   **In particular, student participation is encouraged.**
*   Participants can be a single entrant or a team; there is no restriction on the size of teams.
*   Should participants be related to organizers of the competition, the rest of the committee will decide whether a conflict of interest is at hand. Affected organizers will not be part of the jury for the final evaluation. Please contact the committee chair ([jank@kth.se](mailto:jank@kth.se)) if you are somehow related to one of the organizers. 


### Technical Rules



*   Participants may use any existing software (including closed-source solvers) when solving optimization problems (e.g., LPs and MIPs) as subproblems in the primal heuristic. Participants will be responsible for making the code run on a server and installing the required software packages. Please contact the organizers if you plan on using some more unusual software (typical subsolvers are, e.g., SCIP, IPOPT, and Gurobi). 
*   The code must read the problem in MPS format, and write two types of files as output:
    *   A set of **solution files** logging the best-found solutions that are also reported in the result file. Each file should correspond to a solution and there must be a number appended to the file name in the order that the solutions were found (e.g. SolutionFile1, SolutionFile2, ...). The solution files must be generated at roughly the same time as when the solution is reported in the result file. These solution files will be used to compute the primal integral. It is recommended to only write solutions with the best objective value so far, as writing solution files can take extra time. [Example for one solution file](https://www.mixedinteger.org/2025/competition/SolutionFile1).
    *   A single **result file** logging the elapsed wall time (in seconds with 3 decimals) and the objective value of the best solution found thus far, in the order that the solutions were found. Each row of the result file must correspond to one solution file as described above. At the beginning of the result file, you should also report a starting time for the primal heuristic, which may be the time after the problem has been read and the main algorithm starts (this time will be subtracted when computing the primal integral). [Example](https://www.mixedinteger.org/2025/competition/ResultFile).
*   A feasible solution must satisfy all constraints with a tolerance of 1e-6 and integer feasibility tolerance of 1e-5.
*   The source code may be written in any programming language.
*   For the final evaluation, participants will need to compile and run their code on a Linux server, accessible via SSH.
*   The submissions can use a maximum of 8 threads, no more than 16 GB of RAM, and respect a time limit of 5 minutes (excluding I/O) for each instance in the test set.

In case participants have any doubts about the implementation of specific rules, they should not hesitate to contact the organizers.


### Final Evaluation Criteria

The evaluation will be performed by an expert jury of researchers with experience in computational optimization. They will judge both paper and code submission on two criteria:



1. **Computational performance**: We will evaluate the computational performance based on the [primal integral](https://www.sciencedirect.com/science/article/pii/S0167637713001181). We test the code on a hidden set of test instances that covers instances with different structure and different sources of difficulty. For example, for some instances, it is trivial to find a feasible solution, while for others it can be very challenging to even find one feasible solution.
2. **Novelty and scope**: How innovative is the approach. The minimum requirement for novelty is that the code can’t fully be based on another software. For example, running the commercial solver Gurobi with some specific settings is not novel enough. However, using Gurobi to solve subproblems (for example some sort of decomposition, linearization, or reduced search) is perfectly fine.

The spirit of this competition is to encourage the development of new methods that work in practice. The jury will be free to disqualify submissions that provide no contribution beyond repurposing existing software.


## Submission Requirements


### Registration



*   All participants must register with the full list of team members by sending an e-mail [here](mailto:jankr@kth.se) by January 31th, 2025.
*   Two weeks before the deadline, all teams will receive access to a server for testing installation of their software.
*   Teams of multiple participants must nominate a single contact person to handle the submission of report and code.


### Report

All participants must submit a written report of **10 pages maximum** plus references, in Springer LNCS format. Submissions will be accepted until **March 16, 2025 (AoE)**.

The report must include the following information:



*   A description of the methods developed and implemented, including any necessary citations to the literature and software used.
*   Computational results on the competition testset.
*   Results should include at least the metrics `time to first feasible solution` and `primal integral based on the best-found solution `in aggregated form over the competition test set.
*   Detailed tables (if really needed) can be put in appendix and do not count toward the 10 pages limit.
*   Further analysis of the computational results is welcome.

If the computational work was performed by students only, the participants should include a letter of attestation indicating this.


### Code

The primal heuristic should be executable via a shell script named `PrimHeur.sh` (provided by the participants) which receives arguments on the command line as follows:



1. The first argument is the path in the filesystem to the instance to read, in (gzipped) MPS format.
2. The second argument is the path in the filesystem where the method should write the results (files for solutions found and a result file).

The primal heuristic will thus be executed as


```
sh PrimHeur.sh /path/to/instance.mps.gz /path/to/results
```


with a hard time limit of 5 minutes. Files are specified as absolute paths.
