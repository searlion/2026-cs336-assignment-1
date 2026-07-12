from tokenizer.linear import Linear

layer = Linear(256,256)
print(layer.weight.shape)
print(layer.weight.mean().item(), layer.weight.std().item())