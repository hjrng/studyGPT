import torch
import torch.nn as nn

class Simple(nn.Module):
    def __init__(self,input_dim,output_dim):
        super().__init__()
        self.linaer=nn.Linear(input_dim,output_dim)
        self.layer_norm=nn.LayerNorm(output_dim)

    def forward(self,x):
        x=self.linaer(x)
        print(x)
        x=self.layer_norm(x)
        return x
    
x=torch.randn(2,3)
net=Simple(3,4)
output=net(x)

print(output)


        
     