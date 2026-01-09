#!/usr/bin/env sh
# Simple runner: compiles `main.cpp` (if needed), generates tests and
# runs the program on every input, saving outputs in `outputs/`.
set -e

# Compile if the binary isn't present
if [ ! -f main ]; then
  echo "Compiling main.cpp..."
  g++ -std=c++17 -O2 main.cpp -o main
fi

# Make the test files
python3 scripts/generate_tests.py

mkdir -p outputs

for f in tests/*.in; do
  echo "Running $f"
  ./main < "$f" > outputs/$(basename "$f" .in).out
done

echo "Done — outputs are in ./outputs/"