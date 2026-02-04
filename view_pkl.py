import pickle
import pprint

with open("final_hdfc.pkl", "rb") as f:
    data = pickle.load(f)

# Pretty print so it is readable
pprint.pprint(data, depth=3)
