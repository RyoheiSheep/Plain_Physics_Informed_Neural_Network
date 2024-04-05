import torch
import torch.nn as nn

class FCN(nn.Module):
    def __init__(self,input_size, hidden_size_list,ouput_size):
        super(FCN,self).__init__()
        self.layers=nn.ModuleList()

        self.activation=nn.Tanh()
        #Input Layer

        self.layers.append(nn.Linear(input_size,hidden_size_list[0]))
        self.layers.append(self.activation)

        #Hidden layers
        for i in range(len(hidden_size_list)-1):
            self.layers.append(nn.Linear(hidden_size_list[i],hidden_size_list[i+1]))
            self.layers.append(self.activation)

        self.layers.append(nn.Linear(hidden_size_list[-1],ouput_size))

    def forward(self,x):

        for layer in self.layers:
            x=layer(x)
        return x


if __name__=="__main__":
    net=FCN(2,[4,4,4],1)
    test_input=torch.randn(10,2)
    test_output=net(test_input)

    print(f"net:\n{net}")
    print(f"input:\n{test_input}")
    print(f"output:\n{test_output}")

