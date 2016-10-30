require 'cunn'
require 'nn'

model = torch.load('model/pretrained_model.net')
model = model:float()
torch.save('model/pretrained_model.net.nn', model)
