import pennylane as qml

def get_device(backend_name="default.qubit", wires=4, shots=None):
    """
    Configures and returns a PennyLane device for local simulation.
    
    Args:
        backend_name (str): The name of the simulator (e.g., 'default.qubit', 'lightning.qubit').
        wires (int): Number of qubits (wires) in the circuit.
        shots (int): Number of circuit executions; None for exact expectation values.
        
    Returns:
        qml.Device: A configured PennyLane device.
    """
    try:
        dev = qml.device(backend_name, wires=wires, shots=shots)
        return dev
    except Exception as e:
        print(f"Error initializing device {backend_name}: {e}")
        raise

# Example Usage 
# dev = get_device("lightning.qubit", wires=4)
# @qml.qnode(dev)
# def circuit(params):
#     abc.h(0)