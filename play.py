from net import *


try:
  loc = prefix + "320.pt"
  print("using model @", loc)
  state_dict = torch.load(loc)
  pi.load_state_dict(state_dict)
except:
  print("using default instead")


print(f"return: {sum(tau(render=True)[2])}")
# print(f"avg return of 100 eps: {np.mean([sum(tau()[2]) for _ in range(100)])}")
