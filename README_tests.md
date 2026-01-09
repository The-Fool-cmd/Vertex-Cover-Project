**Tests Overview**
- Location: `tests/` (the generator puts them here)
- To create the test input files, run: `python3 scripts/generate_tests.py`

Each test file `NN.in` follows the input format used in the contest:
- First line: `N M` (number of nodes and edges)
- Next `M` lines: `u v` (0-based node indices, undirected edge between u and v)

What the 25 tests try to cover:
- 01..05: tiny graphs (easy base cases)
- 06: star (center-heavy)
- 07..08: small path and cycle
- 09: complete bipartite K10,10
- 10: mostly a path with a few shortcuts (N=100)
- 11: denser chunk of pairs on N=100
- 12: several disconnected components
- 13: binary-tree layout
- 14: 10x10 grid
- 15: clique (size 15)
- 16: sparse bipartite-like (N=200)
- 17: hub-heavy graph (few high-degree nodes)
- 18: large path (N=1000)
- 19: large star (N=1000)
- 20: deterministic dense-ish chunk (N=500)
- 21: 2x250 grid
- 22: two big cliques joined by one bridge
- 23: empty graph (no edges)
- 24: moderate chunk on N=250
- 25: a forest made of long paths plus a couple of links

Notes:
- The generator is deterministic so you get the same files every run.
- I tried to pick shapes that exercise both correctness and running-time behavior.

Quick run example:
1. Compile: `g++ -std=c++17 -O2 main.cpp -o main`
2. Generate tests: `python3 scripts/generate_tests.py`
3. Run program on all inputs and save outputs:

```sh
mkdir -p outputs
for f in tests/*.in; do
  ./main < "$f" > outputs/$(basename "$f" .in).out
done
```

If you want, I can add a script that compiles and runs everything automatically.
