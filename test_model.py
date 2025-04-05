import pickle
import numpy as np

model_dict = pickle.load(open("model.p", "rb"))
model = model_dict["model"]

# Simulated 21-hand-landmark (42-length) input
fake_input = np.random.rand(42).tolist()

print("Predicting...")
print(model.predict([fake_input]))
