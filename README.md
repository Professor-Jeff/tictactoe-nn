# Tic-Tac-Toe — Three Ways to Build a Player

Companion to [pick-activity](https://github.com/Professor-Jeff/pick-activity) for COMP 138.
The same tiny neural network, written from scratch in NumPy, learns tic-tac-toe
two different ways — and a classic search algorithm plays it a third way.

1. **Minimax** — the deterministic program. Looks ahead and plays perfectly.
2. **Supervised network** — trained on positions that minimax labels. Never sees minimax.
3. **Reinforcement network** — the *same* network, trained only from wins and losses. No labels at all.

## Run it

```bash
pip install numpy
python3 tictactoe_nn.py
```

That trains both networks (about 3 minutes), prints how each does against a
random player and against minimax, then lets you play the reinforcement network.

Options:

```bash
python3 tictactoe_nn.py --mode supervised     # just the supervised network (~35 s)
python3 tictactoe_nn.py --mode reinforce      # just the reinforcement network
python3 tictactoe_nn.py --rounds 3000         # quicker, weaker reinforcement training
python3 tictactoe_nn.py --no-play             # skip the interactive game
```

## The one line that changes

Supervised training pushes the output toward the correct answer:

```python
dz2 = probs - Y                      # predicted minus CORRECT
```

Reinforcement training pushes it toward whatever the network actually did,
scaled by how the game ended (+1 win, -1 loss, 0 draw):

```python
dz2 = G * (probs - chosen)           # predicted minus WHAT I DID, scaled
```

Everything below that line — backpropagation and the weight update — is identical.
