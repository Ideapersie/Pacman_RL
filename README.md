# Pacman MDP Agent

Coursework for the KCL Artificial Intelligence module (MSc AI), built on the UC Berkeley Pacman AI framework (http://ai.berkeley.edu).

`pacman-cw/pacman/mdpAgents.py` treats each Pacman move as a Markov decision process: it builds a reward map from food, capsules and ghost positions, runs value iteration (discount 0.95, up to 100 iterations) over the grid with the non-deterministic movement model the coursework API exposes, and then takes the action with the highest expected utility.

```bash
cd pacman-cw/pacman
python pacman.py -q -n 25 -p MDPAgent -l mediumClassic
```

The Pacman framework is UC Berkeley's. Its licence allows educational use on the condition that solutions are not published, so this repository should stay private.
