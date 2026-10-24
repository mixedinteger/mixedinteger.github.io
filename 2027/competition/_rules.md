## Submission Requirements

### Report

All participants must submit a written report of **10 pages maximum** plus references, in Springer
LNCS format. The report and the code must be submitted together.

The report must include the following information:

- A description of the method developed and implemented, including any necessary citations to the
  literature and software used.
- A section discussing the methodological and/or engineering innovations of the method (see
  [Evaluation Criteria](topic.html#evaluation-criteria)). If you have any clever implementation
  techniques to showcase (e.g., performance optimizations), please highlight them (the jury will not
  check every detail of the code).
- Computational results on the public competition set, including a table of results.

### Code

The code should output a JSON file listing the elements relaxed and kept in the infeasible subsystem
(linear constraints, bound constraints, and integrality constraints). Feasibility certificates can
be optionally specified and count towards certifying the system is irreducible.

Refer to the GitHub repository for the exact
[solution format](https://github.com/mixedinteger/MIPcc27/blob/main/SOLUTION_FORMAT.md) and
[program interface](https://github.com/mixedinteger/MIPcc27/blob/main/PROGRAM_INTERFACE.md).

Public [sample instances and solutions](https://github.com/mixedinteger/MIPcc27/tree/main/smallset)
are available.

## Rules and Eligibility

### Rules for Participation

- Participants must not be an organizer of the competition nor a family member of a competition
  organizer. Otherwise, there is no restriction on participation.
- Student participation is strongly encouraged.
- Should participants be related to organizers of the competition (e.g., students of an organizer),
  the rest of the committee will decide whether a conflict of interest is at hand and affected
  organizers will not be part of the jury for the final evaluation.
- Participants can be a single entrant or a team; there is no restriction on the size of teams.
- Questions may be directed to the competition committee by opening an
  [Issue](https://github.com/mixedinteger/MIPcc27/issues) or
  [Discussion](https://github.com/mixedinteger/MIPcc27/discussions) on the competition GitHub
  repository: <https://github.com/mixedinteger/MIPcc27>, which hosts the
  [solution checker and helper scripts](https://github.com/mixedinteger/MIPcc27#check-your-solutions).

### Registration

Registration is required to participate. Register your team on the [Register page](register.html) by **January 22, 2027 (AoE)**. The registration form includes the
conflict-of-interest disclosure (see Rules for Participation above).

### Technical Rules

- Competitors cannot use commercial solvers as subroutines of the final submission.

### Evaluation Platform

Each instance will have a wall-clock time limit of 3600 seconds, and each run will have at least 8
threads and 32 GB of memory available. The exact hardware will be announced later. See the
[program interface](https://github.com/mixedinteger/MIPcc27/blob/main/PROGRAM_INTERFACE.md) for how
programs are invoked and how time is measured.

## To Be Announced

The following will be published with the final rules:

- Final evaluation details, including how criteria are combined and how failed, timed-out, or
  missing instance results are treated
- Public and hidden evaluation instance sets (sample instances are already available)
- Evaluation platform hardware
- Submission process and exact dates and time zone
