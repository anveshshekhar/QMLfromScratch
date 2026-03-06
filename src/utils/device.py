import pennylane as qml

def get_device(backend_name="default.qubit", wires=4, shots=None):
    try:
        dev = qml.device(backend_name, wires=wires, shots=shots)
        return dev
    except Exception as e:
        print(f"Error initializing device {backend_name}: {e}")
        raise
