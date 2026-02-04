import json
import pickle

# Step 1: Load JSON file
with open("final_hdfc.json", "r", encoding="utf-8") as json_file:
    data = json.load(json_file)

# Step 2: Save as Pickle file
with open("final_hdfc.pkl", "wb") as pkl_file:
    pickle.dump(data, pkl_file)

print("✅ final_hdfc.json successfully converted to final_hdfc.pkl")
