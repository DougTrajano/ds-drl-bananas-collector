# Solving banana collector with Deep Reinforcement Learning

**Author:** Doug Trajano

> [LinkedIn](https://www.linkedin.com/in/douglas-trajano/) - [GitHub](https://github.com/DougTrajano)

The problematic is describe in the video below:

[![](http://i.ytimg.com/vi/heVMs3t9qSk/hqdefault.jpg)](https://www.youtube.com/embed/heVMs3t9qSk)

## 1. Introduction

### How Deep Reinforcement Learning works?

A simple introduction of the concept can be understood in this image:

![](https://wpumacay.github.io/research_blog/imgs/img_rl_loop.png)

The **agent** send an **action** to the **environment**, then, the environment respond with a **state** and a **reward** to the agent.

The objective of the agent is **maximize the cumulative reward**.

## 2. Environment

### How is this environment?

The simulation contains a single agent that navigates a large environment.

**State space**

The state space has `37` dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.

**Reward**

A reward of `+1` is provided for collecting a yellow banana, and a reward of `-1` is provided for collecting a blue banana.

**Actions**

At each time step, it has four actions at its disposal:

- `0` - walk forward 
- `1` - walk backward
- `2` - turn left
- `3` - turn right

## 3. Agent

The agent created on this project consists of an **agent**, a **Deep Q-Learning** and a **memory unit**.

**Agent** `agent.py`

The agent has the methods that interacts with the environment: `step()`, `act()`, `learn()` and some others.

**Deep Q-Learning** `network.py`

The architecture of the model is too simple, it's has an **input layer**, a **hidden layer** and then, a **output layer**.

This neural networks was developed with https://pytorch.org/

**Memory unit** `memory.py`

For the memory unit we have two options that can be used:

- ReplayMemory

> With two simple methods `add()` and `sample()` this memory can storage the experiences and also can returns aleatory some experiences to be used in agent training.

- PrioritizedMemory

> This memory is a little bit more complex, because the `sample()` method doesn't returns the experiences aleatory. It's try to understand how experiences can be used for a better training of the agent applying weights for each experience.

> This memory has two additional methods compared to **RerplayMemory**: `update_probabilities()` and `update_priority()`

> Learn more here: https://arxiv.org/pdf/1511.05952.pdf

## 4. Results

### 4.1 How I found the hyperparameters?

I applied GridSearch to find the best combination of hyperparameters for the agent.

See below the best hyperparameters founded:

```
{
    "state_size": 37,
    "action_size": 4,
    "seed": 199,
    "nb_hidden": [64, 64],
    "learning_rate": 0.001,
    "memory_size": int(1e6),
    "prioritized_memory": true,
    "batch_size": 64,
    "gamma": 0.9,
    "tau": 0.03,
    "small_eps": 0.03,
    "update_every": 4
}
```

### 4.2 The results

**Agent with ReplayMemory**

![](/images/agent_ReplayMemory.png)

**Agent with PrioritizedMemory**

![](/images/agent_PrioritizedMemory.png)

**Agent with PrioritizedMemory losses**

![](/images/agent_PrioritizedMemory_losses.png)


# 5. Testing the agent

See below the agent trained:

![](/images/banana-collector.gif)

# 6. Future work

For the future work I think that we can test a dueling DQN to improve this model. See more about it [here].

Other implementation that we can do is epsilon greedy policy on Prioritized Memory.

(https://arxiv.org/abs/1511.06581).
