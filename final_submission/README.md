# Group 20 RL Project

## Running the code

The main file should be runnable as is once the relevant dependencies are installed (we use PyCharm and Anaconda to automatically detect and enable installing needed packages).

Equally, the supplied `environment.yml` can be used to install the exact conda interpreter used to run the code on our machines. Run `conda env create -f environment.yml` to achieve this.

It is important to note that gym version 0.22.0 needs to be used and not the latest version as otherwise the gym video wrappers does not work.

You will also need to create the following directories in the same root directory as all the other python files:
- A plot directory called `plots`.
- A directory to store the agents final trained networks called `tmp`, containing a `ddpg` and `td3` folder.
- A directory to store videos of the agents performance called `videos`.

## Existing results

Our current best agents can be found under `agents`. These can be loaded using the code in line 60-77 of `main.py`, but when instantiating the agent set the `agent_dir='agents/ddpg'` and `agent_dir='agents/td3'` appropriately.

The hyper-parameters used to train these agents were:
- `actor_learning_rate = 0.001`
- `critic_learning_rate = 0.001`
- `tau = 0.005`
- `gamma = 0.99`
- `layer1_size = 400`
- `layer2_size = 300`
- `max_memory_size = 1,000,000`
- `batch_size = 64`

The results of these agents can be seen in the `final_plots` directory. Equally the log output during training can be found under the `logs` directory.
