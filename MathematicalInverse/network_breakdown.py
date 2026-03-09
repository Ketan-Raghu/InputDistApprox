import torch

# Load the file (map_location handles CPU/GPU differences)
model_data = torch.load('LearnedInverse/mnist_logits_train.pt')

# If it's a state_dict, you can see the layer names
for layer_name, weights in model_data.items():
    print(f"Layer: {layer_name} | Shape: {weights.shape}")
    
    # To see the raw matrix of a specific layer:
    # if "weight" in layer_name:
    #     print(torch.linalg.pinv(weights))
    if "logits" in layer_name:
        print(weights)