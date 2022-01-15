from net import *


try:
  epoch = int(sys.argv[1])
  loc = prefix + f"{epoch}.pt"
  print("using model @", loc)
  state_dict = torch.load(loc)
  pi.load_state_dict(state_dict)
except:
  print("using default instead'")


R = tau(True)[2]
print(f"return: {sum(R)}")
# print(f"avg return of 10 eps: {np.mean([sum(tau()[2]) for _ in range(10)])}")
