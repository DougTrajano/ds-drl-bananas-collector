# Bananas collector

This project contains an agent based on **Deep Reinforcement Learning** that can learn from zero (any labeled data) to collect yellow bananas instead of blue bananas in a vast, square world.

It's use the [Unity Machine Learning Agents Toolkit](https://github.com/Unity-Technologies/ml-agents) to design, train, and evaluate deep reinforcement learning algorithms implementations.

The environment used for this project is the Udacity version of the Banana Collector environment, from Unity. The goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas. 

## Installation

The installation process is divided in three parts:

1. Python 3.6.x
2. Dependencies
3. Unity's Environment

### Python

To install and run this project you will need to have Python 3.6.x.

You could download olds Anaconda versions [here](https://repo.anaconda.com/archive/)

The reason for it is that Unity ml-agents needs TensorFlow in version 1.15 and TensorFlow 1.15 needs Python. 3.6.x.


### Dependencies 

All packages and their versions are describe in [requirements.txt](requirements.txt).

You need to run these commands before to install all dependencies:

```
pip install --upgrade pip
pip install -r requirements.txt
pip -q install ./python
```

You'll have these three commands in the first part of [Navigation.ipynb], as well.

### Unity's Environment

You need only select the environment that matches your operating system:

Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip)
Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86.zip)
Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86_64.zip)

*(For AWS)* If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

Then, place the unzipped folder on the root of this repository.

## License

The contents of this repository are covered under the MIT [License](LICENSE).