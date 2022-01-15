from net import *


# adjust hyperparameters
lr = 2.5e-4
epochs = 600
N = 20  # E[R] ~ sum(pi(a_t | s_t)V_t) / N
mod = 5  # checkpoint once every mod epochs
gamma = 0.99  # reward discount

# init optimiser
opt = torch.optim.Adam(pi.parameters(), lr=lr)

# create files to log epoch losses and rewards
lossesf = open(prefix + f"losses.txt", "w")
rewardsf = open(prefix + f"rewards.txt", "w")

# train network
losses, rewards = train(epochs, N, mod, gamma, opt)

# plot stuff
fig = plt.figure(figsize=(8, 8))

x = list(range(1, epochs + 1))

plt.xlabel("epoch")
plt.ylabel("loss")
plt.plot(x, losses)
plt.savefig(prefix + f"losses.png")

plt.clf()

plt.xlabel("epoch")
plt.ylabel("reward")
plt.plot(x, rewards)
plt.savefig(prefix + f"rewards.png")
