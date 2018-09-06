from __future__ import division, absolute_import, print_function, unicode_literals

import numpy as np

from abc import ABCMeta, abstractmethod

class NodeBase(metaclass=ABCMeta):
    
    def __init__(self):
        self.input_nodes = []
        self.output = None
        self.gradients = []
        
    @abstractmethod
    def forward(self):
        return
    
    #@abstractmethod
    def backward(self):
        return

        
       
class NodeInput(NodeBase):
    def __init__(self, output=None):
        super().__init__() 
        self.output = output
        
    def forward(self):
        pass  # do nothing
        
class NodeAdd(NodeBase):
    def __init__(self, *nodes):
        super().__init__()
        self.input_nodes = nodes[:]  # copy
        self.gradients = [1] * len(nodes)
        
    def forward(self):
        if len(self.input_nodes) == 0:
            raise Exception('Input nodes not set')

        self.output = self.input_nodes[0].output
            
        for i in range(1, len(self.input_nodes)):
            self.output += self.input_nodes[i].output
            
    def backward(self):
        pass  # do nothing, all gradients set to 1 in constructor
                
class NodeMult(NodeBase):
    def __init__(self, nodeA, nodeB):
        super().__init__()
        self.input_nodes = [nodeA, nodeB]
        self.gradients = [None, None]
        
    def forward(self):
        if len(self.input_nodes) != 2:
            raise Exception('Wrong number of input nodes, should be 2')

        self.output = self.input_nodes[0].output * self.input_nodes[1].output
        self.gradients[0] = self.input_nodes[1].output
        self.gradients[1] = self.input_nodes[0].output
            
        for i in range(1, len(self.input_nodes)):
            self.output *= self.input_nodes[i].output
                
class NodePow2(NodeBase):
    def __init__(self, node):
        super().__init__()
        self.input_nodes = [node]
        self.gradients = [None]
        
    def forward(self):
        if len(self.input_nodes) != 1:
            raise Exception('Input nodes not set or too many')
        
        self.output = self.input_nodes[0].output * self.input_nodes[0].output
        self.gradients[0] = 2 * self.input_nodes[0].output
        
    def backward(self):
        if len(self.input_nodes) != 1:
            raise Exception('Input nodes not set or too many')

        self.gradients[0] = 2 * self.input_nodes[0].output
        
                       
def main():

    ni1 = NodeInput( 5.0 )
    ni2 = NodeInput( 5.0 )
    
    ns = NodeAdd(ni1, ni2)
    ns.forward()
    
    nm = NodeMult(ni1, ni2)
    nm.forward()
    
    np = NodePow2(ni1)
    np.forward()
    np.backward()
    
    const = NodeInput( 3.0 )
    x = NodeInput( 1.0 )
    mult = NodeMult( const, x )
    pow = NodePow2( mult )
    
    mult.forward()
    pow.forward()
    
    grad = pow.gradients[0] * mult.gradients[1]
    
    print(grad)
        
if __name__ == '__main__':
    main()