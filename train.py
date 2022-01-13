from net import *


# adjust hyperparameters
lr = 2.5e-4
epochs = 500
N = 20  # E[R] ~ sum(pi(a_t | s_t)V_t) / N
mod = 10  # checkpoint once every mod epochs
gamma = 0.99  # reward discount

opt = torch.optim.Adam(pi.parameters(), lr=lr)

lossesf = open(prefix + "losses.txt", "w")
rewardsf = open(prefix + "rewards.txt", "w")

losses, rewards = train(epochs, N, mod, gamma, opt)

fig = plt.figure(figsize=(8, 8))

x = list(range(1, epochs + 1))

plt.xlabel("epoch")
plt.ylabel("loss")
plt.plot(x, losses)
plt.savefig(prefix + "losses.png")

plt.clf()

plt.xlabel("epoch")
plt.ylabel("reward")
plt.plot(x, rewards)
plt.savefig(prefix + "rewards.png")
