import torch
import torch.nn as nn 
import torch.nn.functional as F 


#####################
### ECOC decoder  ###
#####################
class ecoc_decoder(nn.Module): 
    def __init__(self, W, no_tanh=False): 
        super(ecoc_decoder, self).__init__()
        self.W = W
        self.no_tanh = no_tanh
        self.activation1 = torch.nn.Tanh()   # Tanh to scale each bit in range [-1, 1]

    def forward(self, code):
        W_ = self.W.unsqueeze(0) 
        W_ = W_.repeat(W_.shape[0], 1, 1)
        W_ = torch.transpose(W_, dim0=1, dim1=2)
        if not self.no_tanh:
            code = self.activation1(code).unsqueeze(1)
        else: 
            code = code.unsqueeze(1)
        out = torch.matmul(code, W_.float()).squeeze(1)        
        return out 


###############################
### ECOC Models for CIFAR10 ###
###############################
class ecoc_ensemble_no_bn(nn.Module):  
    def __init__(self, W, num_chunks, num_filter_ens, num_filter_ens_2, num_codes, dataset, activation='tanh'): 
        super(ecoc_ensemble_no_bn, self).__init__()
        self.W = W 
        self.num_chunks = num_chunks
        self.num_codes = num_codes
        self.activation = activation
        self.in_size = 0 # input size
        self.out = 0   # input to fc layer
        if dataset == "CIFAR10": 
            self.in_size = 3 
            self.out = 256
        elif dataset == "Fashion-MNIST": 
            self.in_size = 1
            self.out = 64
        self.models_ensemble = nn.ModuleList(ecoc_no_bn(W, num_chunks, num_filter_ens, num_filter_ens_2, num_codes, self.in_size, self.out, activation) for _ in range(num_chunks))

    def forward(self, x): 
        code = torch.cat([model(x) for model in self.models_ensemble], dim=1)
        return code

class ecoc_no_bn(nn.Module): 
    def __init__(self, W, num_chunks, num_filter_ens, num_filter_ens_2, num_codes, in_size, out, activation): 
        super(ecoc_no_bn, self).__init__()
        self.W = W
        self.activation = activation                # activation function in our case is "tanh"
        self.num_codes = num_codes                  # number of bits in codes. Equivalent to the number of binary classifiers in the ecoc ensemble 
        self.num_chunks = num_chunks                # number of independent models in the ecoc ensemble
        self.num_filter_ens = num_filter_ens        # number of filters in the shared first conv layers 
        self.num_filter_ens_2 = num_filter_ens_2    # number of filters in the independant conv layers at the end of the model
        self.conv2D = torch.nn.Conv2d(in_size, num_filter_ens[0], kernel_size=5, stride=1, padding=2)
        self.conv2D_1 = torch.nn.Conv2d(num_filter_ens[0], num_filter_ens[0], kernel_size=5, stride=1, padding=2)
        self.conv2D_2 = torch.nn.Conv2d(num_filter_ens[0], num_filter_ens[0], kernel_size=3, stride=2, padding=1)
        self.conv2D_3 = torch.nn.Conv2d(num_filter_ens[0], num_filter_ens[1], kernel_size=3, stride=1, padding=1)
        self.conv2D_4 = torch.nn.Conv2d(num_filter_ens[1], num_filter_ens[1], kernel_size=3, stride=1, padding=1)
        self.conv2D_5 = torch.nn.Conv2d(num_filter_ens[1], num_filter_ens[1], kernel_size=3, stride=2, padding=1)
        self.conv2D_6 = torch.nn.Conv2d(num_filter_ens[1], num_filter_ens[2], kernel_size=3, stride=1, padding=1)
        self.conv2D_7 = torch.nn.Conv2d(num_filter_ens[2], num_filter_ens[2], kernel_size=3, stride=1, padding=1)
        self.conv2D_8 = torch.nn.Conv2d(num_filter_ens[2], num_filter_ens[2], kernel_size=3, stride=2, padding=1)
        self.independent_heads = nn.ModuleList(bloc_out_no_bn(num_filter_ens, num_filter_ens_2, out, num_codes) for _ in range(int(self.num_codes/self.num_chunks)))

    def forward(self, x): 
        x = F.relu(self.conv2D(x))
        x = F.relu(self.conv2D_1(x))
        x = F.relu(self.conv2D_2(x))
        x = F.relu(self.conv2D_3(x))
        x = F.relu(self.conv2D_4(x))
        x = F.relu(self.conv2D_5(x))
        x = F.relu(self.conv2D_6(x)) 
        x = F.relu(self.conv2D_7(x))
        x = F.relu(self.conv2D_8(x))
        code = torch.cat([out_head(x) for out_head in self.independent_heads], dim=1)   
        return code 

class bloc_out_no_bn(nn.Module): 
    def __init__(self,num_filter_ens, num_filter_ens_2, out, num_codes): 
        super(bloc_out_no_bn, self).__init__()
        self.conv2D_12 = torch.nn.Conv2d(num_filter_ens[2], num_filter_ens_2[0], kernel_size=2, stride=1, padding=1)    
        self.conv2D_13 = torch.nn.Conv2d(num_filter_ens_2[0], num_filter_ens_2[0], kernel_size=2, stride=1, padding=0) ## C'est le workaround que j'ai trouvé pour que la size en sortie de la couche soit de []
        self.dense_1 = torch.nn.Linear(out, 1) 

    def forward(self, x): 
        output = F.relu(self.conv2D_12(x))
        output = F.relu(self.conv2D_13(output))
        output = output.flatten(1)
        output = self.dense_1(output)
        return output




############################################
#### SIMPLE baseline vanilla network net ###
############################################

class simple(nn.Module): 
    def __init__(self, num_filter_ens, num_filter_ens_2, dataset): 
        super(simple, self).__init__()
        self.num_filter_ens = num_filter_ens        # number of filters in the shared first conv layers 
        self.num_filter_ens_2 = num_filter_ens_2    # number of filters in the independant conv layers at the end of the model
        if dataset == "CIFAR10": 
            in_size = 3 
            out = 256
        elif dataset == "Fashion-MNIST": 
            in_size = 1
            out = 64
        self.conv2D = torch.nn.Conv2d(in_size, num_filter_ens[0], kernel_size=5, stride=1, padding=2)
        self.conv2D_1 = torch.nn.Conv2d(num_filter_ens[0], num_filter_ens[0], kernel_size=5, stride=1, padding=2)
        self.conv2D_2 = torch.nn.Conv2d(num_filter_ens[0], num_filter_ens[0], kernel_size=3, stride=2, padding=1)
        self.conv2D_3 = torch.nn.Conv2d(num_filter_ens[0], num_filter_ens[1], kernel_size=3, stride=1, padding=1)
        self.conv2D_4 = torch.nn.Conv2d(num_filter_ens[1], num_filter_ens[1], kernel_size=3, stride=1, padding=1)
        self.conv2D_5 = torch.nn.Conv2d(num_filter_ens[1], num_filter_ens[1], kernel_size=3, stride=2, padding=1)
        self.conv2D_6 = torch.nn.Conv2d(num_filter_ens[1], num_filter_ens[2], kernel_size=3, stride=1, padding=1)
        self.conv2D_7 = torch.nn.Conv2d(num_filter_ens[2], num_filter_ens[2], kernel_size=3, stride=1, padding=1)
        self.conv2D_8 = torch.nn.Conv2d(num_filter_ens[2], num_filter_ens[2], kernel_size=3, stride=2, padding=1)
        self.conv2D_12 = torch.nn.Conv2d(num_filter_ens[2], num_filter_ens_2[0], kernel_size=2, stride=1, padding=1)    
        self.conv2D_13 = torch.nn.Conv2d(num_filter_ens_2[0], num_filter_ens_2[0], kernel_size=2, stride=1, padding=0) ## C'est le workaround que j'ai trouvé pour que la size en sortie de la couche soit de []
        self.dense_1 = torch.nn.Linear(out, 10) 

    def forward(self, x): 
        x = F.relu(self.conv2D(x))
        x = F.relu(self.conv2D_1(x))
        x = F.relu(self.conv2D_2(x))
        x = F.relu(self.conv2D_3(x))
        x = F.relu(self.conv2D_4(x))
        x = F.relu(self.conv2D_5(x))
        x = F.relu(self.conv2D_6(x)) 
        x = F.relu(self.conv2D_7(x))
        x = F.relu(self.conv2D_8(x))
        output = F.relu(self.conv2D_12(x))
        output = F.relu(self.conv2D_13(output))
        output = output.flatten(1)
        output = self.dense_1(output)
        
        return output 

class simple_ensemble(nn.Module): 
    def __init__(self, model_list, no_sotfmax=False): 
        super(simple_ensemble, self).__init__()
        self.models = model_list
        self.no_softmax = no_sotfmax
        self.activation = torch.nn.Softmax(dim=2) 
      
    def forward(self, x): 
        out = torch.stack([model(x) for model in self.models], dim=1)
        if not self.no_softmax:  
            out = self.activation(out)
        out = torch.sum(out, dim=1)    
        return out




