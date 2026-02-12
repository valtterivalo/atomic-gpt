# atomic-gpt

karpathy's [microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) ported to C. same algorithm, no dependencies, ~95x faster.

trains a tiny character-level GPT on a names dataset and generates new ones.

## benchmark (500 training steps, M-series Mac)

|                | Python   | C       |
|----------------|----------|---------|
| training       | ~37s     | ~0.25s  |
| inference (x20)| included | 0.27ms  |
| final loss     | ~2.1     | ~2.6    |

losses differ because the RNGs differ (mersenne twister vs xorshift64), not because of a bug. both converge.

## run

```
make data   # downloads names.txt
make gpt    # compiles
./gpt       # trains + generates
```

or the python original:

```
python3 gpt.py
```

full side-by-side:

```
make benchmark
```

## what changed from the python

- fixed a buffer overflow in tokenization
- added a raw-double inference path (skips autograd when we don't need gradients)
- O(1) char lookup instead of linear scan
- adam correction factors hoisted out of the inner loop

## credit

original python implementation: [microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) by [andrej karpathy](https://github.com/karpathy).
