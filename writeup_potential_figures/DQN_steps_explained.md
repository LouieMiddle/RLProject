# DQN

I have listed the steps involved in a deep Q-network (DQN) below:

- Preprocess and feed the game screen (state s) to our DQN, which will return the Q-values of all possible actions in the state
- Select an action using the epsilon-greedy policy. With the probability epsilon, we select a random action a and with probability 1-epsilon, we select an action that has a maximum Q-value, such as a = argmax(Q(s,a,w))
- Perform this action in a state s and move to a new state s’ to receive a reward. This state s’ is the preprocessed image of the next game screen. We store this transition in our replay buffer as <s,a,r,s’>
- Next, sample some random batches of transitions from the replay buffer and calculate the loss
- It is known that: which is just the squared difference between target Q and predicted Q
- Perform gradient descent with respect to our actual network parameters in order to minimize this loss
- After every C iterations, copy our actual network weights to the target network weights
- Repeat these steps for M number of episodes
