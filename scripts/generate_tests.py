#!/usr/bin/env python3
"""
A small script I wrote to spit out 25 test inputs for the vertex-cover
problem. Each file goes into `tests/` and is named `01.in`..`25.in`.

File format: first line `N M`, then `M` lines `u v` with 0-based node ids.

I included a mix of cases: tiny examples, paths, cycles, cliques, stars,
some trees/grids, bipartite examples, a couple of hub-like graphs and a
few larger instances so you can check performance. It's deterministic so
it's easy to reproduce the same set later.

Run: `python3 scripts/generate_tests.py`
"""
from pathlib import Path
from itertools import combinations

OUTDIR = Path(__file__).resolve().parent.parent / "tests"
OUTDIR.mkdir(exist_ok=True)

def write_test(name, n, edges):
    path = OUTDIR / name
    with path.open("w") as f:
        f.write(f"{n} {len(edges)}\n")
        for u,v in edges:
            f.write(f"{u} {v}\n")
    print(f"Wrote {path} (n={n}, m={len(edges)})")

# 01: one node, no edges
write_test("01.in", 1, [])

# 02: two nodes with one edge
write_test("02.in", 2, [(0,1)])

# 03: tiny path (3 nodes)
write_test("03.in", 3, [(0,1),(1,2)])

# 04: small cycle (4 nodes)
write_test("04.in", 4, [(0,1),(1,2),(2,3),(3,0)])

# 05: complete graph K5 (all pairs)
n=5
edges=[(i,j) for i in range(n) for j in range(i+1,n)]
write_test("05.in", n, edges)

# 06: small star (center 0, 9 leaves)
n=10
edges=[(0,i) for i in range(1,n)]
write_test("06.in", n, edges)

# 07: path of length 20 (simple long-ish chain)
n=20
edges=[(i,i+1) for i in range(n-1)]
write_test("07.in", n, edges)

# 08: 20-node cycle
n=20
edges=[(i,(i+1)%n) for i in range(n)]
write_test("08.in", n, edges)

# 09: complete bipartite K10,10 (left 0..9, right 10..19)
n=20
edges=[(i,j) for i in range(0,10) for j in range(10,20)]
write_test("09.in", n, edges)

# 10: mostly a path (N=100) with a few short chords added
n=100
edges=[(i,i+1) for i in range(n-1)]
# add a few (i, i+2) shortcuts to make it slightly less trivial
for i in range(0,21):
    edges.append((i,i+2))
write_test("10.in", n, edges)

# 11: denser-ish graph on 100 nodes — first 2000 pairs in lexicographic order
n=100
edges=[]
count=2000
for i in range(n):
    for j in range(i+1,n):
        edges.append((i,j))
        if len(edges)>=count:
            break
    if len(edges)>=count:
        break
write_test("11.in", n, edges)

# 12: three disconnected components in one file:
#   - path of 5 nodes (0..4)
#   - cycle of 7 nodes (5..11)
#   - star with center 12 and leaves 13..19
edges=[]
edges += [(i,i+1) for i in range(0,4)]
edges += [(i,i+1) for i in range(5,11)]
edges.append((11,5))
edges += [(12,i) for i in range(13,20)]
write_test("12.in", 20, edges)

# 13: a binary tree layout on 50 nodes (parent = (i-1)//2)
n=50
edges=[((i-1)//2, i) for i in range(1,n)]
write_test("13.in", n, edges)

# 14: 10x10 grid (nice for locality tests)
rows,cols = 10,10
n = rows*cols
edges=[]
for r in range(rows):
    for c in range(cols):
        u = r*cols + c
        if c+1<cols:
            edges.append((u, r*cols + (c+1)))
        if r+1<rows:
            edges.append((u, (r+1)*cols + c))
write_test("14.in", n, edges)

# 15: clique K15 (all pairs)
n=15
edges=[(i,j) for i in range(n) for j in range(i+1,n)]
write_test("15.in", n, edges)

# 16: sparse bipartite-ish graph (N=200, ~150 edges) — deterministic spread
n=200
edges=[]
for idx in range(150):
    left = idx % 100
    right = 100 + ((idx*7) % 100)
    edges.append((left, right))
# remove duplicates but keep the order
seen=set()
uniq=[]
for u,v in edges:
    if (u,v) not in seen:
        uniq.append((u,v)); seen.add((u,v))
write_test("16.in", n, uniq)

# 17: hub-heavy graph: a few hubs (0,1,2) connected to most nodes
n=100
hubs=[0,1,2]
edges=[]
for h in hubs:
    for v in range(3,n):
        edges.append((h,v))
# link the hubs together too
for a,b in combinations(hubs,2):
    edges.append((a,b))
write_test("17.in", n, edges)

# 18: big path with 1000 nodes (tests linear behavior)
n=1000
edges=[(i,i+1) for i in range(n-1)]
write_test("18.in", n, edges)

# 19: big star (center 0, 999 leaves)
n=1000
edges=[(0,i) for i in range(1,n)]
write_test("19.in", n, edges)

# 20: deterministic "random-like" chunk (first 2500 pairs) on 500 nodes
n=500
edges=[]
count=2500
for i in range(n):
    for j in range(i+1,n):
        edges.append((i,j))
        if len(edges)>=count:
            break
    if len(edges)>=count:
        break
write_test("20.in", n, edges)

# 21: 2x250 grid (thin rectangle)
rows,cols=2,250
n=rows*cols
edges=[]
for r in range(rows):
    for c in range(cols):
        u = r*cols + c
        if c+1<cols:
            edges.append((u, r*cols + (c+1)))
        if r+1<rows:
            edges.append((u, (r+1)*cols + c))
write_test("21.in", n, edges)

# 22: two big cliques (0..99 and 100..199) with a single bridge (0-100)
n=200
edges=[]
for i in range(0,100):
    for j in range(i+1,100):
        edges.append((i,j))
for i in range(100,200):
    for j in range(i+1,200):
        edges.append((i,j))
edges.append((0,100))
write_test("22.in", n, edges)

# 23: empty graph (no edges) — useful to check edge cases
write_test("23.in", 30, [])

# 24: moderate deterministic chunk (1000 pairs) on 250 nodes
n=250
edges=[]
count=1000
for i in range(n):
    for j in range(i+1,n):
        edges.append((i,j))
        if len(edges)>=count:
            break
    if len(edges)>=count:
        break
write_test("24.in", n, edges)

# 25: small forest built from four long paths plus a couple of cross links
n=200
edges=[]
start=0
for k in range(4):
    length = 50
    for i in range(start, start+length-1):
        edges.append((i, i+1))
    start += length
edges.append((0,50))
edges.append((100,150))
write_test("25.in", n, edges)

print('Done. Tests written to `tests/`.')
