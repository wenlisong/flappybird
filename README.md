# Flappy Bird

* My English and Coding ability is limited. If there is any mistakes, please correct me.
I will try to make this project better.

## Overview
In this project, I try using Reinforcement Learning to play a game, called "FLAPPY BIRD".
[yenchenlin](https://github.com/yenchenlin) had provided a version and some idea.
BUT what I want to do is a little different (or say, I play it in a different way).
I make the pipe move **UP and DOWN** each frame. If I can play the game under this situation, 
I estimate that it play the origin game very well.

## Reproduction
First, I'll reproduce what [yenchenlin/DeepLearningFlappyBird](https://github.com/yenchenlin/DeepLearningFlappyBird)
did in a clearer and more flexible structure. Of course, the main idea and methods are not changed. In this procedure,
I fix some bugs, record the score per episode using TensorBoard, visualize how images(game screen) processed and make
it can run in the "dummy"(headless) mode(say, running without video devices).

### How to Run?
```
git clone https://github.com/wenlisong/flappybird.git
cd flappybird
python run_dqn.py
```


## Installation Dependencies:
* python 3.6
* tensorflow 1.8.0
* opencv-python
* pygame

