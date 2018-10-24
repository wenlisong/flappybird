# Flappy Bird

* My English and Coding ability is limited. If there is any mistakes, please correct me.
I will try to make this project better.

## Overview
In this project, I try using Reinforcement Learning to play a game, called "FLAPPY BIRD".
[yenchenlin](https://github.com/yenchenlin) had provided a version and some idea.
BUT what I want to do is a little different (or say, I play it in a different way).
I make the pipe move **UP and DOWN** each frame. If I can play the game under this situation, 
I estimate that it play the origin game very well.

## Introduction
### Game
"Flappy Bird" is a side-scrolling game in which the gameplay action is viewed from a side-view camera angle, 
and the onscreen characters can generally only move to the left or right. It is just like "Super Mario Bros". 
In the game, there is a flying bird, named "Faby", who moves continuously towards the right. 
The objective is to direct Faby between sets of Mario-like pipes. If Faby hits the pipe, then player lose. 
Faby briefly flaps upward each time that the player click left mouse button; if the buttuon is not clicked, 
Faby falls because of gravity. Each pair of pipes that he navigates between earns the player one point.
Therefore, player should fly Faby as far as possible to get high score.
<img src="./assets/readme/flappybird.png">
### Reinforcement Learning


## Reproduction
First, I'll reproduce what [yenchenlin/DeepLearningFlappyBird](https://github.com/yenchenlin/DeepLearningFlappyBird)
did in a clearer and more flexible structure. Of course, the main idea and methods are not changed. In this procedure,
I fix some bugs, record the score per episode using TensorBoard, visualize how images(game screen) processed and make
it can run in the "dummy"(headless) mode(say, running without video devices).

### How to Run?
```
git clone https://github.com/wenlisong/flappybird.git
cd flappybird
python main.py -g fb -n dqn
```


## Installation Dependencies:
* python 3.6
* tensorflow 1.8.0
* opencv-python
* pygame

