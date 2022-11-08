## Overview

Writeup of the assignment: https://docs.google.com/document/d/1xISBzRIknxd7BiY62SYsrsz262WtZtttBz4i1c-d0WQ/edit?usp=sharing

## Installation

We copied over the base tetris package from [Gym-SimplifiedTetris](https://github.com/OliverOverend/gym-simplifiedtetris). 

Create a conda environment and install the dependencies by
```bash
pip install -r requirements.txt
```

## Developing code

The main script to run and evaluate policies is `evaluate_policy.py`. 

```bash
python evaluate_policy.py --render
```

The script by default runs a random policy. The render flag shows a window with the tetris board.

If you remove the render flag, the script runs the policy 10 times and prints the score

```bash
python evaluate_policy.py --render
```

which should return `Summary over 10 epsiodes:  Highest score: 0.0 Mean score: 0.0 Std score: 0.0`
Not surprising that the random policy scores 0! 


We also provide a baseline heuristic policy. To run that:
```bash
python evaluate_policy.py  --policy heuristic --render 
```

This is a game played by a heuristic hand designed by Pierre Dellacherie which is unreasonably effective! In fact, it is so good, you may never see it actually lose a game. To benchmark it, you may want to run it without the render flag
```bash
python evaluate_policy.py  --policy heuristic 
```

You should see it reach the max score in all 10 episodes (i.e. last the whole of 10000 steps). 


Finally, **YOU** will have to implement `def test_policy(obs, env):` which will call your trained policy. Once you do so, you can evaluate your policy via

```bash
python evaluate_policy.py  --policy test --render 
```

## Acknowledgement

We copy gym_simplifiedtetris from the excellent package [Gym-SimplifiedTetris](https://github.com/OliverOverend/gym-simplifiedtetris). 

