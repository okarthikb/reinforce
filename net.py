import gym
import time
import tqdm
import torch
import collections
import numpy as np
import matplotlib.pyplot as plt


# change env name for different env
env_name = "Pong-v0"
env = gym.make(env_name)
actions = [2, 3]  # custom list or [i for i in range(env.action_space.n)]
prefix = "checkpoints/"  # directory to save logs


# change network architecture for diff env
class net(torch.nn.Module):
  def __init__(self):
    super(net, self).__init__()
    self.fc = torch.nn.Linear(1600, 128)
    self.out = torch.nn.Linear(128, len(actions))

  def forward(self, x):
    x = torch.nn.functional.relu(self.fc(x))
    # always softmax output
    x = torch.nn.functional.softmax(self.out(x), -1)
    return x


pi = net()

frames = 4  # num of consecutive frames to process


def tensor(x):
  return torch.as_tensor(x, dtype=torch.float32)


def phi(seq):
  seq = np.array(seq)
  seq = seq[::3, 34:194, :, 1]
  seq[seq == 72] = 0
  seq[seq != 0] = 1
  seq = seq[1] - seq[0]
  seq = seq[::4, ::4]
  return seq.ravel()


def logits(s):
  return torch.distributions.Categorical(pi(s))


def act(s):
  return logits(s).sample().item()


def loss(S, A, V):
  logp = logits(S).log_prob(A)
  return -(logp * V).sum()


# generate a trajectory
def tau(render=False):
  done = False
  # states, actions, and rewards
  S, A, R = [], [], []
  s = env.reset()
  # store current (and previous) frame(s)
  seq = collections.deque(
    [np.zeros_like(s)] * (frames - 1) + [s], 
    maxlen=frames
  )
  with torch.no_grad():
    while not done:
      s = phi(seq)  # pre-process frames
      S.append(s)  # store processed
      i = act(tensor(s))  # sample action index
      a = actions[i]  # get action
      s, r, done, _ = env.step(a)  # take action
      seq.append(s)  # append next frame to seq
      A.append(i)  # store action
      R.append(r)  # store reward
      if render:
        env.render()
        time.sleep(1 / 45)
  # close env and return transitions
  env.close()
  return np.array(S), np.array(A), np.array(R)


def train(epochs, N, mod, gamma, opt):
  losses = []
  rewards = []
  lossesf = open(prefix + "losses.txt", "w")
  rewardsf = open(prefix + "rewards.txt", "w")

  start = time.time()

  for epoch in range(1, epochs + 1):
    J = 0  # reward
    r = 0  # avg return for N episodes
    # calculated expected reward
    for _ in range(N):
      S, A, R = tau()
      r += R.sum()
      for t in range(len(R) - 2, -1, -1):
        R[t] += gamma * R[t + 1]
      R = (R - R.mean()) / R.std()
      J += loss(tensor(S), tensor(A), tensor(R))
    # backprop and step
    opt.zero_grad()
    J /= N
    J.backward()
    opt.step()
    # return reward and loss
    rewards.append(r / N)
    losses.append(abs(J.item()))
    # boring logging :P
    if epoch % mod == 0:
      print("epoch: {}  reward: {:.3f}  loss: {:.3f}".format(epoch, rewards[-1], losses[-1]))
      lossesf.write(f"{losses[-1]}\n")
      rewardsf.write(f"{rewards[-1]}\n")
      torch.save(pi.state_dict(), prefix + f"{epoch}.pt")

  print("train time: {:.2f}s".format(time.time() - start))

  return losses, rewards
