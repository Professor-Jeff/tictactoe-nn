"""
Tic-Tac-Toe — three ways.

1. The deterministic way: minimax, a search that plays perfectly by looking
   ahead. This is the "rules" program, the same role the if/else played in
   pick_activity_nn.py.
2. The supervised way: the SAME tiny network from pick_activity_nn.py, trained
   on (board, best move) examples that minimax labels. It never sees minimax.
3. The reinforcement way: the SAME network again, but now nobody labels
   anything. It plays games, and every move it made gets pushed toward
   "do that again" if it won and away if it lost.

The network in (2) and (3) is identical:  9 inputs -> 128 hidden -> 9 outputs.
Only where the training signal comes from changes.

    SUPERVISED     dz2 = probs - Y              (predicted minus CORRECT)
    REINFORCE      dz2 = G * (probs - chosen)   (predicted minus WHAT I DID,
                                                 scaled by how the game ended)

Board encoding, from the point of view of whoever moves next:
    +1 = my piece     -1 = opponent's piece     0 = empty
This lets one network play as either X or O.

Requires only NumPy:  pip install numpy
Run:                  python tictactoe_nn.py            (train both, evaluate, play)
                      python tictactoe_nn.py --mode supervised
                      python tictactoe_nn.py --mode reinforce
                      python tictactoe_nn.py --no-play  (skip the interactive game)
                      python tictactoe_nn.py --rounds 3000   (quicker, weaker REINFORCE)

What to expect (seeds 42 and 7, Apple M-series laptop, ~3 min total):
    supervised   ~35 s     ~98% optimal moves on positions it never trained on,
                           never loses to minimax as X, loses ~10% as O
    reinforce    ~2.5 min  ~75% optimal moves, never loses to minimax as X,
                           draws every game as O on most seeds (seed 7: loses 13%)

Two things worth noticing in those numbers. The supervised net picks an
optimal move 99% of the time and STILL drops games as O, because its rare
mistakes sit in exactly the forced positions where O has one saving reply.
The reinforcement net is "only" 75% optimal yet never loses as X, because it
never cared which of several equally good moves it picked — it only learned
not to lose, in the positions it actually reached. Different training
signals teach different things, from the same network.
"""

import argparse
import functools
import time

import numpy as np

X_, O_ = 1, 2                      # player ids. X always moves first.
LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8),      # rows
         (0, 3, 6), (1, 4, 7), (2, 5, 8),      # columns
         (0, 4, 8), (2, 4, 6)]                 # diagonals


# ---------------------------------------------------------------------------
# PART 1: the game itself and the deterministic player (minimax)
# ---------------------------------------------------------------------------

def winner(board):
    """Returns 1 or 2 if that player has three in a row, else 0."""
    for a, b, c in LINES:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a]
    return 0


def legal_moves(board):
    return [i for i in range(9) if board[i] == 0]


def game_over(board):
    return winner(board) != 0 or not legal_moves(board)


def other(player):
    return O_ if player == X_ else X_


@functools.lru_cache(maxsize=None)
def minimax(board, player):
    """Look ahead to the end of the game and return (score, best_moves).
    score is from `player`'s point of view: +1 win, 0 draw, -1 loss.
    best_moves is the tuple of every move that achieves that score — there
    are often several equally good ones (e.g. all four corners to open).
    The board must be a tuple so results can be cached."""
    w = winner(board)
    if w:
        return (1 if w == player else -1), ()
    moves = legal_moves(board)
    if not moves:
        return 0, ()

    best, best_moves = -2, []
    for m in moves:
        child = board[:m] + (player,) + board[m + 1:]
        score = -minimax(child, other(player))[0]      # my score = minus theirs
        if score > best:
            best, best_moves = score, [m]
        elif score == best:
            best_moves.append(m)
    return best, tuple(best_moves)


def minimax_move(board, player, rng):
    """A perfect player. Picks at random among the equally-best moves so games
    against it are not all identical."""
    _, best = minimax(tuple(board), player)
    return int(rng.choice(best))


def random_move(board, player, rng):
    return int(rng.choice(legal_moves(board)))


# ---------------------------------------------------------------------------
# PART 2: turning a board into network inputs
# ---------------------------------------------------------------------------

def encode(board, player):
    """9 numbers: +1 where `player` has a piece, -1 for the opponent, 0 empty."""
    return np.array([1.0 if c == player else (-1.0 if c else 0.0) for c in board])


def legal_mask(board):
    """1.0 for empty squares, 0.0 for occupied. Used to forbid illegal moves."""
    return np.array([1.0 if c == 0 else 0.0 for c in board])


def all_positions():
    """Every board that can actually arise in play, with whose turn it is.
    Tic-tac-toe is small: 5,478 legal positions, 4,520 of which still need a
    move. The supervised network trains on all of them."""
    seen, out = set(), []

    def walk(board, player):
        if (board, player) in seen or game_over(board):
            return
        seen.add((board, player))
        out.append((board, player))
        for m in legal_moves(board):
            walk(board[:m] + (player,) + board[m + 1:], other(player))

    walk((0,) * 9, X_)
    return out


# ---------------------------------------------------------------------------
# PART 3: the network — copied from pick_activity_nn.py, three constants changed
#   9 inputs  ->  hidden layer (128 neurons)  ->  9 outputs
# ---------------------------------------------------------------------------

N_IN, N_HID, N_OUT = 9, 128, 9


def initialize_weights(rng):
    """Random starting weights. Nothing here 'knows' anything yet."""
    return {
        "W1": rng.normal(0, 0.3, size=(N_IN, N_HID)),
        "b1": np.zeros(N_HID),
        "W2": rng.normal(0, 0.3, size=(N_HID, N_OUT)),
        "b2": np.zeros(N_OUT),
    }


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(float)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def forward(params, X, mask):
    """Run boards through the network. `mask` is 1 for legal squares, 0 for
    occupied ones; occupied squares get their score pushed to -infinity so
    their probability is exactly 0 and they never receive gradient."""
    z1 = X @ params["W1"] + params["b1"]
    a1 = relu(z1)
    z2 = a1 @ params["W2"] + params["b2"]
    z2 = np.where(mask > 0, z2, -1e9)
    probs = softmax(z2)
    cache = {"X": X, "z1": z1, "a1": a1}
    return probs, cache


def cross_entropy_loss(probs, Y):
    return -np.sum(Y * np.log(probs + 1e-12)) / len(Y)


def backward(params, cache, dz2):
    """Backpropagation, unchanged from pick_activity_nn.py except that the
    caller hands us dz2 — the output-layer error — because supervised and
    reinforcement training compute it differently. Everything from here down
    is the same chain rule."""
    dW2 = cache["a1"].T @ dz2
    db2 = dz2.sum(axis=0)
    da1 = dz2 @ params["W2"].T
    dz1 = da1 * relu_derivative(cache["z1"])
    dW1 = cache["X"].T @ dz1
    db1 = dz1.sum(axis=0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def update_weights(params, grads, learning_rate):
    for key in params:
        params[key] -= learning_rate * grads[key]


def network_move(params, board, player, rng=None):
    """Use the network as a player. With rng=None it picks the most likely
    square (greedy); with an rng it samples, which is how it explores."""
    probs, _ = forward(params, encode(board, player)[None, :], legal_mask(board)[None, :])
    probs = probs[0]
    if rng is None:
        return int(probs.argmax())
    return int(rng.choice(9, p=probs / probs.sum()))


# ---------------------------------------------------------------------------
# PART 4a: SUPERVISED training — minimax labels every position
# ---------------------------------------------------------------------------

def make_supervised_dataset():
    """For every position, the target spreads probability evenly over all of
    minimax's equally-best moves. The network never sees minimax itself."""
    X, M, Y = [], [], []
    for board, player in all_positions():
        _, best = minimax(board, player)
        target = np.zeros(9)
        target[list(best)] = 1.0 / len(best)
        X.append(encode(board, player))
        M.append(legal_mask(board))
        Y.append(target)
    return np.array(X), np.array(M), np.array(Y)


def is_optimal(probs, Y):
    """Did the network's top choice land on one of the equally-best moves?"""
    return Y[np.arange(len(Y)), probs.argmax(axis=1)] > 0


def train_supervised(params, X, M, Y, epochs, learning_rate, log_every):
    """Exactly the loop from pick_activity_nn.py."""
    for epoch in range(1, epochs + 1):
        probs, cache = forward(params, X, M)
        loss = cross_entropy_loss(probs, Y)
        dz2 = (probs - Y) / len(Y)                      # predicted minus correct
        grads = backward(params, cache, dz2)
        update_weights(params, grads, learning_rate)
        if epoch == 1 or epoch % log_every == 0:
            print(f"  epoch {epoch:5d}   loss {loss:.4f}   optimal-move rate {is_optimal(probs, Y).mean():.1%}")
    return params


# ---------------------------------------------------------------------------
# PART 4b: REINFORCEMENT training — no labels, just outcomes
# ---------------------------------------------------------------------------

def play_game(params, opponent, net_plays, rng):
    """Play one game. The network samples its moves (exploration). Returns the
    winner and a list of (inputs, mask, move) for every move the NETWORK made.
    opponent=None means self-play: the network plays both sides and every
    move is recorded, each from its own mover's point of view."""
    board, player, history = [0] * 9, X_, []
    while not game_over(board):
        if opponent is None or player == net_plays:
            x, m = encode(board, player), legal_mask(board)
            move = network_move(params, board, player, rng)
            history.append((x, m, move, player))
        else:
            move = opponent(board, player, rng)
        board[move] = player
        player = other(player)
    return winner(board), history


def reward_for(mover, result):
    """+1 if the mover won, -1 if lost, 0 for a draw."""
    if result == 0:
        return 0.0
    return 1.0 if result == mover else -1.0


def train_reinforce(params, rounds, games_per_round, learning_rate, rng, log_every):
    """REINFORCE. Each round: play a batch of games, then one weight update.
    The output-layer error for each recorded move is
        dz2 = G * (probs - one_hot(move))
    i.e. the supervised formula with the label replaced by the move the
    network actually chose, scaled by G (how the game ended for that mover).
    Won: push toward every move you made. Lost: push away. Draw: no change.

    Opponents are mixed. Pure self-play can settle into a policy that beats
    itself but never learned to block; a few games against a random player
    and against minimax keep it honest.

    The learning rate shrinks linearly to a tenth of its starting value: big
    steps early while everything is wrong, small steps late so the noisy
    outcome signal stops knocking a decent policy around."""
    schedule = [None] * 6 + [random_move] * 2 + [minimax_move] * 2    # 60/20/20
    for rnd in range(1, rounds + 1):
        lr = learning_rate * (1 - 0.9 * (rnd - 1) / rounds)
        Xs, Ms, moves, Gs = [], [], [], []
        tally = {"win": 0, "draw": 0, "loss": 0}
        for g in range(games_per_round):
            opponent = schedule[g % len(schedule)]
            net_plays = X_ if g % 2 == 0 else O_
            result, history = play_game(params, opponent, net_plays, rng)
            for x, m, move, mover in history:
                G = reward_for(mover, result)
                if opponent is minimax_move and result == 0:
                    G = 0.5          # a draw is the best you can do vs perfect play
                Xs.append(x); Ms.append(m); moves.append(move); Gs.append(G)
            if opponent is not None:
                key = "draw" if result == 0 else ("win" if result == net_plays else "loss")
                tally[key] += 1

        X, M, G = np.array(Xs), np.array(Ms), np.array(Gs)[:, None]
        chosen = np.zeros((len(moves), 9))
        chosen[np.arange(len(moves)), moves] = 1.0

        probs, cache = forward(params, X, M)
        dz2 = G * (probs - chosen) / len(moves)        # predicted minus what I did, scaled
        grads = backward(params, cache, dz2)
        update_weights(params, grads, lr)

        if rnd == 1 or rnd % log_every == 0:
            n = sum(tally.values())
            print(f"  round {rnd:5d}   vs opponents this round: "
                  f"won {tally['win']/n:.0%}  drew {tally['draw']/n:.0%}  lost {tally['loss']/n:.0%}")
    return params


# ---------------------------------------------------------------------------
# PART 5: measuring a player — win/draw/loss against random and minimax
# ---------------------------------------------------------------------------

def evaluate(params, opponent, games, rng, net_plays):
    """The network plays greedily (no exploration) as `net_plays` (X or O)."""
    tally = {"win": 0, "draw": 0, "loss": 0}
    for g in range(games):
        board, player = [0] * 9, X_
        while not game_over(board):
            if player == net_plays:
                move = network_move(params, board, player)
            else:
                move = opponent(board, player, rng)
            board[move] = player
            player = other(player)
        result = winner(board)
        tally["draw" if result == 0 else ("win" if result == net_plays else "loss")] += 1
    return tally


def report(label, params, rng, X, M, Y, games=100):
    """Win/draw/loss as X and as O against each opponent, plus how often the
    network's top choice is one of minimax's best moves over ALL positions.
    Playing O against minimax is the hard case: X has the initiative, and O
    has to find the one or two drawing replies every time."""
    probs, _ = forward(params, X, M)
    print(f"  {label:<12} optimal move in {is_optimal(probs, Y).mean():.0%} of all {len(X)} positions")
    for name, opp in [("random", random_move), ("minimax", minimax_move)]:
        for side, who in [(X_, "as X"), (O_, "as O")]:
            t = evaluate(params, opp, games, rng, side)
            print(f"  {'':<12} vs {name:<8}{who}:  won {t['win']/games:4.0%}  drew {t['draw']/games:4.0%}  lost {t['loss']/games:4.0%}")


# ---------------------------------------------------------------------------
# PART 6: play against it
# ---------------------------------------------------------------------------

def show(board, probs=None):
    sym = {0: ".", X_: "X", O_: "O"}
    for r in range(3):
        row = " ".join(sym[board[3 * r + c]] for c in range(3))
        if probs is not None:
            row += "      " + " ".join(f"{probs[3 * r + c]:4.0%}" if board[3 * r + c] == 0 else "  - " for c in range(3))
        print("   " + row)
    print()


def play_human(params):
    print("Squares are numbered 1-9, left to right, top to bottom.")
    human = X_ if input("Play as X (first) or O (second)? [X/o] ").strip().lower() != "o" else O_
    board, player = [0] * 9, X_
    while not game_over(board):
        probs, _ = forward(params, encode(board, player)[None, :], legal_mask(board)[None, :])
        if player == human:
            show(board)
            try:
                move = int(input("Your move (1-9): ")) - 1
            except ValueError:
                print("Please type a number from 1 to 9.")
                continue
            if move not in legal_moves(board):
                print("That square is taken or out of range.")
                continue
        else:
            move = int(probs[0].argmax())
            print(f"Network plays square {move + 1}. Its probabilities for each empty square:")
            show(board, probs[0])
        board[move] = player
        player = other(player)
    show(board)
    w = winner(board)
    print("You win!" if w == human else ("Draw." if w == 0 else "The network wins."))


# ---------------------------------------------------------------------------
# PART 7: put it together
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["both", "supervised", "reinforce"], default="both")
    ap.add_argument("--no-play", action="store_true", help="skip the interactive game")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rounds", type=int, default=8000, help="REINFORCE training rounds (40 games each)")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    trained = {}
    X, M, Y = make_supervised_dataset()       # every position + minimax's best moves

    if args.mode in ("both", "supervised"):
        print("\n=== SUPERVISED: learn to imitate minimax from labeled positions ===")
        # Hold out 20% of positions the network never trains on, to see whether
        # it generalizes or just memorized a lookup table.
        idx = rng.permutation(len(X))
        split = int(0.8 * len(X))
        tr, te = idx[:split], idx[split:]
        print(f"{len(X)} positions: training on {len(tr)}, holding out {len(te)}")

        params = initialize_weights(rng)
        t0 = time.time()
        train_supervised(params, X[tr], M[tr], Y[tr], epochs=4000, learning_rate=0.5, log_every=500)
        probs, _ = forward(params, X[te], M[te])
        print(f"Optimal-move rate on {len(te)} held-out positions: {is_optimal(probs, Y[te]).mean():.1%}"
              f"   ({time.time() - t0:.0f}s)")
        trained["supervised"] = params

    if args.mode in ("both", "reinforce"):
        print("\n=== REINFORCE: learn from wins and losses, no labels ===")
        params = initialize_weights(rng)
        t0 = time.time()
        train_reinforce(params, rounds=args.rounds, games_per_round=40, learning_rate=0.3, rng=rng,
                        log_every=max(1, args.rounds // 10))
        print(f"   ({time.time() - t0:.0f}s)")
        trained["reinforce"] = params

    print("\n=== Evaluation: 100 games per line, network plays greedily (no exploration) ===")
    untrained = initialize_weights(rng)
    report("untrained", untrained, rng, X, M, Y)
    for label, params in trained.items():
        report(label, params, rng, X, M, Y)

    if not args.no_play:
        label = "reinforce" if "reinforce" in trained else "supervised"
        print(f"\n=== Play against the {label} network ===")
        while True:
            play_human(trained[label])
            if input("Again? [y/N] ").strip().lower() != "y":
                break


if __name__ == "__main__":
    main()
