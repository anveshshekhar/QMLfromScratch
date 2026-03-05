import pennylane as qml

class TwoLocalAnsatz:
    def __init__(self, num_layers, num_wires):
        self.num_layers = num_layers
        self.num_wires = num_wires

    def apply(self, params):
        expected_params = self.num_layers * self.num_wires
        if len(params) != expected_params:
            raise ValueError(f"Expected {expected_params} parameters, but got {len(params)}.")
            
        paramx = 0
        
        for layer in range(self.num_layers):
            for wire in range(self.num_wires):
                qml.RY(params[paramx], wires=wire)
                paramx += 1
            
            for wire in range(self.num_wires):
                qml.CNOT(wires=[wire, (wire + 1) % self.num_wires])