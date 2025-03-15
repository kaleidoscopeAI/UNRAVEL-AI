error = {"error": "Not found"}
                    self2.wfile.write(json.dumps(error).encode())
                    as sp
            def do_POST(self2):ifft
                content_length = int(self2.headers['Content-Length']) Any
                post_data = self2.rfile.read(content_length).decode('utf-8')
                futures import ThreadPoolExecutor, ProcessPoolExecutor
                try:
                    request = json.loads(post_data)
                except json.JSONDecodeError:
                    self2.send_response(400)
                    self2.send_header('Content-Type', 'application/json')
                    self2.end_headers()
                    
                    error = {"error": "Invalid JSON"}
                    self2.wfile.write(json.dumps(error).encode())
                    returno
                     abstractmethod
                if 'operation' not in request:
                    self2.send_response(400)
                    self2.send_header('Content-Type', 'application/json')
                    self2.end_headers() format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                    Logger(__name__)
                    error = {"error": "Missing 'operation' parameter"}
                    self2.wfile.write(json.dumps(error).encode())=============
                    return
                    ==========================================================
                operation = request['operation']
                params = request.get('params', {})
                (self, n_qubits: int):
                # Handle the API requestth n qubits."""
                response = self.api.handle_request(operation, params)
                te_dim = 2 ** n_qubits
                self2.send_response(200)state_dim, dtype=np.complex128)
                self2.send_header('Content-Type', 'application/json')
                self2.end_headers()
                te(self, gate_matrix: np.ndarray, target_qubits: List[int]):
        """Apply a quantum gate to specified qubits."""
        # Calculate the full transformation matrix
        full_matrix = self._expand_gate(gate_matrix, target_qubits)
        # Apply the gate
        self.amplitudes = full_matrix @ self.amplitudes
        # Normalize to handle floating-point errors
        self.normalize()
        
    def _expand_gate(self, gate_matrix: np.ndarray, target_qubits: List[int]) -> np.ndarray:
        """Expand the gate matrix to act on the full Hilbert space."""
        # Implementation of tensor product expansion for quantum gates
        sorted_targets = sorted(target_qubits)
        n_targets = len(sorted_targets)
        
        # Check if the gate matrix matches the number of target qubits
        if gate_matrix.shape != (2**n_targets, 2**n_targets):
            raise ValueError(f"Gate matrix shape {gate_matrix.shape} doesn't match for {n_targets} qubits")
            
        # Build the full matrix using sparse matrices for efficiency
        indices = list(range(self.n_qubits))
        for i, target in enumerate(sorted_targets):
            indices.remove(target)
        
        # Permute qubits to bring targets to the beginning
        permutation = sorted_targets + indices
        inv_permutation = [0] * self.n_qubits
        for i, p in enumerate(permutation):
            inv_permutation[p] = i
            
        # Calculate permutation matrices
        perm = np.zeros((self.state_dim, self.state_dim), dtype=np.complex128)
        inv_perm = np.zeros((self.state_dim, self.state_dim), dtype=np.complex128)
        
        for i in range(self.state_dim):
            # Convert i to binary, permute bits, convert back to decimal
            bin_i = format(i, f'0{self.n_qubits}b')
            perm_bits = ''.join(bin_i[inv_permutation[j]] for j in range(self.n_qubits))
            perm_i = int(perm_bits, 2)
            perm[perm_i, i] = 1
            inv_perm[i, perm_i] = 1
            
        # Create the expanded gate
        expanded_gate = np.identity(2**(self.n_qubits - n_targets), dtype=np.complex128)
        expanded_gate = np.kron(gate_matrix, expanded_gate)
        
        # Apply the permutations
        return inv_perm @ expanded_gate @ perm
    
    def normalize(self):
        """Normalize the quantum state."""
        norm = np.sqrt(np.sum(np.abs(self.amplitudes) ** 2))
        if norm > 1e-10:  # Avoid division by near-zero
            self.amplitudes /= norm
            
    def measure(self, collapse: bool = True) -> Tuple[int, float]:
        """Measure the quantum state, optionally collapsing it."""
        probabilities = np.abs(self.amplitudes) ** 2
        result = np.random.choice(self.state_dim, p=probabilities)
        
        if collapse:
            # Collapse the state to the measured basis state
            self.amplitudes = np.zeros_like(self.amplitudes)
            self.amplitudes[result] = 1.0
            
        # Return the measurement result and its probability
        return result, probabilities[result]
    
    def measure_qubit(self, qubit_index: int, collapse: bool = True) -> Tuple[int, float]:
        """Measure a specific qubit."""
        if qubit_index >= self.n_qubits:
            raise ValueError(f"Qubit index {qubit_index} out of range for {self.n_qubits} qubits")
            
        # Calculate probabilities for qubit being 0 or 1
        prob_0 = 0.0
        prob_1 = 0.0
        
        for i in range(self.state_dim):
            # Check if the qubit_index bit is 0 or 1
            if (i >> qubit_index) & 1 == 0:
                prob_0 += np.abs(self.amplitudes[i]) ** 2
            else:
                prob_1 += np.abs(self.amplitudes[i]) ** 2
                
        # Normalize probabilities (in case of floating-point errors)
        total_prob = prob_0 + prob_1
        prob_0 /= total_prob
        prob_1 /= total_prob
        
        # Determine the result
        if np.random.random() < prob_0:
            result = 0
            prob = prob_0
        else:
            result = 1
            prob = prob_1
            
        if collapse:
            # Collapse the state
            new_amplitudes = np.zeros_like(self.amplitudes)
            normalization = 0.0
            
            for i in range(self.state_dim):
                bit_val = (i >> qubit_index) & 1
                if bit_val == result:
                    new_amplitudes[i] = self.amplitudes[i]
                    normalization += np.abs(self.amplitudes[i]) ** 2
                    
            new_amplitudes /= np.sqrt(normalization)
            self.amplitudes = new_amplitudes
            
        return result, prob
        
    def entangle(self, other_state: 'QuantumState') -> 'QuantumState':
        """Entangle this quantum state with another one."""
        total_qubits = self.n_qubits + other_state.n_qubits
        entangled = QuantumState(total_qubits)
        
        # Tensor product of states
        entangled.amplitudes = np.kron(self.amplitudes, other_state.amplitudes)
        return entangled

    def density_matrix(self) -> np.ndarray:
        """Calculate the density matrix representation of the state."""
        return np.outer(self.amplitudes, np.conjugate(self.amplitudes))
    
    def partial_trace(self, keep_qubits: List[int]) -> np.ndarray:
        """Perform a partial trace, keeping only specified qubits."""
        trace_qubits = [i for i in range(self.n_qubits) if i not in keep_qubits]
        
        # Calculate dimensions
        keep_dim = 2 ** len(keep_qubits)
        trace_dim = 2 ** len(trace_qubits)
        
        # Initialize reduced density matrix
        reduced_dm = np.zeros((keep_dim, keep_dim), dtype=np.complex128)
        
        # Convert to density matrix
        full_dm = self.density_matrix()
        
        # Perform partial trace
        for i in range(trace_dim):
            bin_i = format(i, f'0{len(trace_qubits)}b')
            
            for j in range(keep_dim):
                for k in range(keep_dim):
                    # Calculate full indices
                    idx_j = self._combine_indices(j, i, keep_qubits, trace_qubits)
                    idx_k = self._combine_indices(k, i, keep_qubits, trace_qubits)
                    
                    # Add to reduced density matrix
                    reduced_dm[j, k] += full_dm[idx_j, idx_k]
                    
        return reduced_dm
    
    def _combine_indices(self, keep_idx: int, trace_idx: int, 
                         keep_qubits: List[int], trace_qubits: List[int]) -> int:
        """Combine separated indices into a full Hilbert space index."""
        keep_bits = format(keep_idx, f'0{len(keep_qubits)}b')
        trace_bits = format(trace_idx, f'0{len(trace_qubits)}b')
        
        # Combine bits
        full_bits = ['0'] * self.n_qubits
        for i, qubit in enumerate(keep_qubits):
            full_bits[qubit] = keep_bits[i]
        for i, qubit in enumerate(trace_qubits):
            full_bits[qubit] = trace_bits[i]
            
        # Convert to decimal
        return int(''.join(full_bits), 2)

class QuantumGates:
    """Common quantum gates."""
    
    @staticmethod
    def I() -> np.ndarray:
        """Identity gate."""
        return np.array([[1, 0], [0, 1]], dtype=np.complex128)
        
    @staticmethod
    def X() -> np.ndarray:
        """Pauli-X (NOT) gate."""
        return np.array([[0, 1], [1, 0]], dtype=np.complex128)
        
    @staticmethod
    def Y() -> np.ndarray:
        """Pauli-Y gate."""
        return np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        
    @staticmethod
    def Z() -> np.ndarray:
        """Pauli-Z gate."""
        return np.array([[1, 0], [0, -1]], dtype=np.complex128)
        
    @staticmethod
    def H() -> np.ndarray:
        """Hadamard gate."""
        return np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
        
    @staticmethod
    def S() -> np.ndarray:
        """Phase gate."""
        return np.array([[1, 0], [0, 1j]], dtype=np.complex128)
        
    @staticmethod
    def T() -> np.ndarray:
        """T gate (π/8 gate)."""
        return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)
        
    @staticmethod
    def CNOT() -> np.ndarray:
        """Controlled-NOT gate."""
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=np.complex128)
        
    @staticmethod
    def SWAP() -> np.ndarray:
        """SWAP gate."""
        return np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.complex128)
    
    @staticmethod
    def Rx(theta: float) -> np.ndarray:
        """Rotation around X-axis."""
        return np.array([
            [np.cos(theta/2), -1j * np.sin(theta/2)],
            [-1j * np.sin(theta/2), np.cos(theta/2)]
        ], dtype=np.complex128)
        
    @staticmethod
    def Ry(theta: float) -> np.ndarray:
        """Rotation around Y-axis."""
        return np.array([
            [np.cos(theta/2), -np.sin(theta/2)],
            [np.sin(theta/2), np.cos(theta/2)]
        ], dtype=np.complex128)
        
    @staticmethod
    def Rz(theta: float) -> np.ndarray:
        """Rotation around Z-axis."""
        return np.array([
            [np.exp(-1j * theta/2), 0],
            [0, np.exp(1j * theta/2)]
        ], dtype=np.complex128)
        
    @staticmethod
    def CZ() -> np.ndarray:
        """Controlled-Z gate."""
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1]
        ], dtype=np.complex128)
        
    @staticmethod
    def Toffoli() -> np.ndarray:
        """Toffoli (CCNOT) gate."""
        toffoli = np.eye(8, dtype=np.complex128)
        # Swap the last two elements of the last two rows
        toffoli[6, 6] = 0
        toffoli[6, 7] = 1
        toffoli[7, 6] = 1
        toffoli[7, 7] = 0
        return toffoli

class QuantumRegister:
    """A quantum register composed of qubits."""
    
    def __init__(self, n_qubits: int):
        """Initialize a quantum register with n qubits."""
        self.state = QuantumState(n_qubits)
        self.n_qubits = n_qubits
        
    def apply_gate(self, gate: Union[str, np.ndarray], 
                 target_qubits: List[int], 
                 params: Optional[List[float]] = None):
        """Apply a gate to target qubits."""
        # Get the gate matrix
        if isinstance(gate, str):
            gate_matrix = self._get_gate_matrix(gate, params)
        else:
            gate_matrix = gate
            
        # Apply the gate
        self.state.apply_gate(gate_matrix, target_qubits)
        
    def _get_gate_matrix(self, gate_name: str, params: Optional[List[float]] = None) -> np.ndarray:
        """Get the gate matrix by name."""
        gates = QuantumGates()
        
        if gate_name == 'I':
            return gates.I()
        elif gate_name == 'X':
            return gates.X()
        elif gate_name == 'Y':
            return gates.Y()
        elif gate_name == 'Z':
            return gates.Z()
        elif gate_name == 'H':
            return gates.H()
        elif gate_name == 'S':
            return gates.S()
        elif gate_name == 'T':
            return gates.T()
        elif gate_name == 'CNOT':
            return gates.CNOT()
        elif gate_name == 'SWAP':
            return gates.SWAP()
        elif gate_name == 'CZ':
            return gates.CZ()
        elif gate_name == 'Toffoli':
            return gates.Toffoli()
        elif gate_name == 'Rx':
            if params is None or len(params) < 1:
                raise ValueError("Rx gate requires a theta parameter")
            return gates.Rx(params[0])
        elif gate_name == 'Ry':
            if params is None or len(params) < 1:
                raise ValueError("Ry gate requires a theta parameter")
            return gates.Ry(params[0])
        elif gate_name == 'Rz':
            if params is None or len(params) < 1:
                raise ValueError("Rz gate requires a theta parameter")
            return gates.Rz(params[0])
        else:
            raise ValueError(f"Unknown gate: {gate_name}")
            
    def measure(self, collapse: bool = True) -> Tuple[int, float]:
        """Measure the entire register."""
        return self.state.measure(collapse)
        
    def measure_qubit(self, qubit_index: int, collapse: bool = True) -> Tuple[int, float]:
        """Measure a specific qubit."""
        return self.state.measure_qubit(qubit_index, collapse)
        
    def get_probabilities(self) -> np.ndarray:
        """Get measurement probabilities for all states."""
        return np.abs(self.state.amplitudes) ** 2
        
    def get_statevector(self) -> np.ndarray:
        """Get the state vector."""
        return self.state.amplitudes.copy()
        
    def set_statevector(self, statevector: np.ndarray):
        """Set the state vector directly."""
        if len(statevector) != self.state.state_dim:
            raise ValueError(f"State vector dimension {len(statevector)} doesn't match register dimension {self.state.state_dim}")
        self.state.amplitudes = statevector.copy()
        self.state.normalize()
        
    def reset(self):
        """Reset the register to |0...0⟩ state."""
        self.state.amplitudes = np.zeros_like(self.state.amplitudes)
        self.state.amplitudes[0] = 1.0

class QuantumCircuit:
    """A quantum circuit for gate-based quantum computation."""
    
    def __init__(self, n_qubits: int):
        """Initialize a quantum circuit with n qubits."""
        self.n_qubits = n_qubits
        self.register = QuantumRegister(n_qubits)
        self.operations = []  # List of operations to execute
        
    def add_gate(self, gate: str, target_qubits: List[int], params: Optional[List[float]] = None):
        """Add a gate to the circuit."""
        self.operations.append(('gate', gate, target_qubits, params))
        return self
        
    def add_measurement(self, target_qubits: List[int]):
        """Add a measurement operation."""
        self.operations.append(('measure', target_qubits))
        return self
        
    def reset(self):
        """Reset the circuit's register and clear operations."""
        self.register.reset()
        self.operations = []
        return self
        
    def x(self, qubit: int):
        """Apply X gate to a qubit."""
        return self.add_gate('X', [qubit])
        
    def y(self, qubit: int):
        """Apply Y gate to a qubit."""
        return self.add_gate('Y', [qubit])
        
    def z(self, qubit: int):
        """Apply Z gate to a qubit."""
        return self.add_gate('Z', [qubit])
        
    def h(self, qubit: int):
        """Apply Hadamard gate to a qubit."""
        return self.add_gate('H', [qubit])
        
    def s(self, qubit: int):
        """Apply S gate to a qubit."""
        return self.add_gate('S', [qubit])
        
    def t(self, qubit: int):
        """Apply T gate to a qubit."""
        return self.add_gate('T', [qubit])
        
    def rx(self, qubit: int, theta: float):
        """Apply Rx gate to a qubit."""
        return self.add_gate('Rx', [qubit], [theta])
        
    def ry(self, qubit: int, theta: float):
        """Apply Ry gate to a qubit."""
        return self.add_gate('Ry', [qubit], [theta])
        
    def rz(self, qubit: int, theta: float):
        """Apply Rz gate to a qubit."""
        return self.add_gate('Rz', [qubit], [theta])
        
    def cnot(self, control: int, target: int):
        """Apply CNOT gate between control and target qubits."""
        return self.add_gate('CNOT', [control, target])
        
    def swap(self, qubit1: int, qubit2: int):
        """Apply SWAP gate between two qubits."""
        return self.add_gate('SWAP', [qubit1, qubit2])
        
    def cz(self, control: int, target: int):
        """Apply CZ gate between control and target qubits."""
        return self.add_gate('CZ', [control, target])
        
    def toffoli(self, control1: int, control2: int, target: int):
        """Apply Toffoli (CCNOT) gate."""
        return self.add_gate('Toffoli', [control1, control2, target])
        
    def barrier(self):
        """Add a barrier (no operation, just for visualization)."""
        self.operations.append(('barrier',))
        return self
        
    def run(self, shots: int = 1) -> Dict[str, int]:
        """Run the circuit for a specified number of shots."""
        # Reset the register
        self.register.reset()
        
        # Dictionary to store measurement results
        results = {}
        
        for _ in range(shots):
            # Reset for each shot
            self.register.reset()
            
            # Execute all operations
            measurement_results = {}
            
            for op in self.operations:
                if op[0] == 'gate':
                    _, gate, target_qubits, params = op
                    self.register.apply_gate(gate, target_qubits, params)
                elif op[0] == 'measure':
                    _, target_qubits = op
                    for qubit in target_qubits:
                        result, _ = self.register.measure_qubit(qubit)
                        measurement_results[qubit] = result
                        
            # Format the result as a binary string
            if measurement_results:
                sorted_qubits = sorted(measurement_results.keys())
                result_str = ''.join(str(measurement_results[q]) for q in sorted_qubits)
                
                if result_str in results:
                    results[result_str] += 1
                else:
                    results[result_str] = 1
                    
        # Convert counts to probabilities
        probabilities = {k: v / shots for k, v in results.items()}
        return probabilities
        
    def get_statevector(self) -> np.ndarray:
        """Get the final state vector after circuit execution (without measurements)."""
        # Reset the register
        self.register.reset()
        
        # Execute all gate operations (skip measurements)
        for op in self.operations:
            if op[0] == 'gate':
                _, gate, target_qubits, params = op
                self.register.apply_gate(gate, target_qubits, params)
                    
        return self.register.get_statevector()
        
    def depth(self) -> int:
        """Calculate the circuit depth."""
        depth = 0
        qubit_layers = [-1] * self.n_qubits
        
        for op in self.operations:
            if op[0] == 'gate':
                _, _, target_qubits, _ = op
                
                # Find the most recent layer among target qubits
                latest_layer = max(qubit_layers[q] for q in target_qubits)
                
                # Assign this operation to the next layer
                new_layer = latest_layer + 1
                
                # Update depth if needed
                depth = max(depth, new_layer + 1)
                
                # Update the layer for all target qubits
                for q in target_qubits:
                    qubit_layers[q] = new_layer
            elif op[0] == 'barrier':
                # Synchronize all qubits
                max_layer = max(qubit_layers)
                qubit_layers = [max_layer] * self.n_qubits
                
        return depth
        
    def to_matrix(self) -> np.ndarray:
        """Convert the circuit to a unitary matrix (without measurements)."""
        # Start with identity matrix
        dim = 2 ** self.n_qubits
        matrix = np.eye(dim, dtype=np.complex128)
        
        # Apply each gate operation in reverse order (matrix multiplication order)
        for op in reversed(self.operations):
            if op[0] == 'gate':
                _, gate, target_qubits, params = op
                
                # Get the gate matrix
                gate_matrix = self.register._get_gate_matrix(gate, params)
                
                # Expand the gate to act on the full Hilbert space
                expanded_gate = QuantumState(self.n_qubits)._expand_gate(gate_matrix, target_qubits)
                
                # Apply the gate
                matrix = expanded_gate @ matrix
                
        return matrix

class QuantumSimulator:
    """Quantum simulator for quantum-inspired computation."""
    
    def __init__(self, max_qubits: int = 20):
        """Initialize the quantum simulator with a maximum number of qubits."""
        self.max_qubits = max_qubits
        
    def create_circuit(self, n_qubits: int) -> QuantumCircuit:
        """Create a new quantum circuit."""
        if n_qubits > self.max_qubits:
            raise ValueError(f"Number of qubits {n_qubits} exceeds maximum {self.max_qubits}")
        return QuantumCircuit(n_qubits)
        
    def run_circuit(self, circuit: QuantumCircuit, shots: int = 1024) -> Dict[str, int]:
        """Run a quantum circuit for the specified number of shots."""
        return circuit.run(shots)
        
    def get_statevector(self, circuit: QuantumCircuit) -> np.ndarray:
        """Get the statevector from a circuit."""
        return circuit.get_statevector()
        
    def get_unitary(self, circuit: QuantumCircuit) -> np.ndarray:
        """Get the unitary matrix representing the circuit."""
        return circuit.to_matrix()
        
    def quantum_fourier_transform(self, n_qubits: int) -> QuantumCircuit:
        """Create a Quantum Fourier Transform circuit."""
        qft = QuantumCircuit(n_qubits)
        
        for i in range(n_qubits):
            # Apply Hadamard
            qft.h(i)
            
            # Apply controlled rotations
            for j in range(i + 1, n_qubits):
                theta = 2 * np.pi / (2 ** (j - i))
                qft.add_gate('Rz', [j], [theta])
                
        # Swap qubits (bit reversal)
        for i in range(n_qubits // 2):
            qft.swap(i, n_qubits - i - 1)
            
        return qft
        
    def simulate_grover(self, n_qubits: int, oracle_function: Callable[[int], bool], 
                       iterations: Optional[int] = None) -> QuantumCircuit:
        """
        Simulate Grover's algorithm with a given oracle function.
        
        Args:
            n_qubits: Number of qubits in the search space
            oracle_function: Function that returns True for target items
            iterations: Number of Grover iterations (defaults to optimal)
            
        Returns:
            The circuit after execution
        """
        # Create the circuit
        circuit = QuantumCircuit(n_qubits)
        
        # Step 1: Initialize in superposition
        for i in range(n_qubits):
            circuit.h(i)
            
        # Calculate optimal number of iterations if not specified
        N = 2 ** n_qubits
        if iterations is None:
            iterations = int(np.floor(np.pi/4 * np.sqrt(N)))
            
        # Implement oracle as a diagonal matrix
        oracle_matrix = np.eye(N, dtype=np.complex128)
        for i in range(N):
            if oracle_function(i):
                oracle_matrix[i, i] = -1  # Phase flip for marked items
                
        # Diffusion operator as a matrix
        diffusion = np.full((N, N), 2/N, dtype=np.complex128) - np.eye(N, dtype=np.complex128)
        
        # Step 2: Apply Grover iterations
        for _ in range(iterations):
            # Oracle
            circuit.add_gate(oracle_matrix, list(range(n_qubits)))
            
            # Diffusion
            circuit.add_gate(diffusion, list(range(n_qubits)))
            
        return circuit
        
    def simulate_quantum_phase_estimation(self, 
                                        n_counting_qubits: int,
                                        unitary_matrix: np.ndarray) -> QuantumCircuit:
        """
        Simulate Quantum Phase Estimation algorithm.
        
        Args:
            n_counting_qubits: Number of qubits for the counting register
            unitary_matrix: Unitary matrix whose eigenvalue we want to estimate
            
        Returns:
            The circuit after execution
        """
        # Determine the size of the unitary matrix
        unitary_size = unitary_matrix.shape[0]
        n_target_qubits = int(np.log2(unitary_size))
        
        if 2**n_target_qubits != unitary_size:
            raise ValueError(f"Unitary matrix size {unitary_size} is not a power of 2")
            
        total_qubits = n_counting_qubits + n_target_qubits
        circuit = QuantumCircuit(total_qubits)
        
        # Initialize counting qubits in superposition
        for i in range(n_counting_qubits):
            circuit.h(i)
            
        # Apply controlled-U operations
        for i in range(n_counting_qubits):
            # Create controlled version of U^(2^i)
            power = 2 ** i
            repeated_unitary = np.linalg.matrix_power(unitary_matrix, power)
            
            # Convert to controlled operation
            controlled_u = np.eye(2 * unitary_size, dtype=np.complex128)
            controlled_u[unitary_size:, unitary_size:] = repeated_unitary
            
            # Apply controlled operation
            control_qubit = n_counting_qubits - 1 - i  # Reversed order
            target_qubits = list(range(n_counting_qubits, total_qubits))
            circuit.add_gate(controlled_u, [control_qubit] + target_qubits)
            
        # Apply inverse QFT to counting qubits
        inverse_qft = self.quantum_fourier_transform(n_counting_qubits)
        inverse_qft_matrix = inverse_qft.to_matrix().conj().T  # Hermitian conjugate for inverse
        
        counting_qubits = list(range(n_counting_qubits))
        circuit.add_gate(inverse_qft_matrix, counting_qubits)
        
        return circuit
    
    def simulate_shor(self, N: int, a: int) -> Dict[str, Any]:
        """
        Simulate Shor's factoring algorithm for factoring N using coprime a.
        
        Args:
            N: The number to factor
            a: A coprime to N (gcd(a, N) = 1)
            
        Returns:
            Dictionary with results of the simulation
        """
        import math
        
        # Check if a and N are coprime
        if math.gcd(a, N) != 1:
            raise ValueError(f"a={a} and N={N} are not coprime")
            
        # Determine the number of qubits needed
        n_count = 2 * math.ceil(math.log2(N))  # Counting register
        n_work = math.ceil(math.log2(N))       # Work register
        
        # Create circuit and registers
        circuit = QuantumCircuit(n_count + n_work)
        
        # Initialize counting register in superposition
        for i in range(n_count):
            circuit.h(i)
            
        # Initialize work register to |1⟩
        # (skip, as |0...01⟩ requires no operations starting from |0...0⟩)
        work_start = n_count
        circuit.x(work_start)  # Set the least significant bit to 1
        
        # Apply modular exponentiation: |x⟩|1⟩ → |x⟩|a^x mod N⟩
        # This is a complex operation typically simulated classically
        # For each x from 0 to 2^n_count - 1:
        x_dim = 2**n_count
        N_dim = 2**n_work
        
        modexp_matrix = np.zeros((x_dim * N_dim, x_dim * N_dim), dtype=np.complex128)
        
        for x in range(x_dim):
            ax_mod_N = pow(a, x, N)
            
            # For each input state |x⟩|y⟩
            for y in range(N_dim):
                # Map to |x⟩|(y * a^x) mod N⟩
                y_new = (y * ax_mod_N) % N_dim
                
                # Calculate indices in the full state space
                idx_in = x * N_dim + y
                idx_out = x * N_dim + y_new
                
                modexp_matrix[idx_out, idx_in] = 1
                
        # Apply the modular exponentiation unitary
        all_qubits = list(range(n_count + n_work))
        circuit.add_gate(modexp_matrix, all_qubits)
        
        # Apply inverse QFT to the counting register
        inverse_qft = self.quantum_fourier_transform(n_count)
        inverse_qft_matrix = inverse_qft.to_matrix().conj().T
        
        counting_qubits = list(range(n_count))
        circuit.add_gate(inverse_qft_matrix, counting_qubits)
        
        # Measure the counting register
        circuit.add_measurement(counting_qubits)
        
        # Run the circuit
        results = circuit.run(shots=1000)
        
        # Post-process results to find the period
        periods = []
        
        for result_str, count in results.items():
            # Convert binary string to integer
            measured_value = int(result_str, 2)
            
            # Convert to a fraction using continued fractions
            fraction = self._continued_fraction_expansion(measured_value, 2**n_count)
            
            if fraction[1] < N and fraction[1] > 1:
                periods.append(fraction[1])
                
        # Find factors using the periods
        factors = set()
        
        for r in periods:
            # If r is even, compute gcd(a^(r/2) ± 1, N)
            if r % 2 == 0:
                x = pow(a, r // 2, N)
                factor1 = math.gcd(x + 1, N)
                factor2 = math.gcd(x - 1, N)
                
                if 1 < factor1 < N:
                    factors.add(factor1)
                if 1 < factor2 < N:
                    factors.add(factor2)
                    
        return {
            "factors": list(factors),
            "periods": periods,
            "measurements": results
        }
        
    def _continued_fraction_expansion(self, numerator: int, denominator: int) -> Tuple[int, int]:
        """
        Find the continued fraction expansion of numerator/denominator.
        Returns the closest convergent as (numerator, denominator).
        """
        import math
        
        a = numerator
        b = denominator
        convergents = []
        
        while b:
            convergents.append(a // b)
            a, b = b, a % b
            
        # Calculate the convergents
        n = [1, convergents[0]]
        d = [0, 1]
        
        for i in range(2, len(convergents)):
            n.append(convergents[i-1] * n[i-1] + n[i-2])
            d.append(convergents[i-1] * d[i-1] + d[i-2])
            
        return (n[-1], d[-1])
        
    def simulate_vqe(self, 
                   hamiltonian: np.ndarray, 
                   ansatz: Callable[[QuantumCircuit, List[float]], None],
                   initial_params: List[float],
                   optimizer: Callable[[Callable[[List[float]], float], List[float]], Tuple[List[float], float]],
                   n_qubits: int) -> Dict[str, Any]:
        """
        Simulate the Variational Quantum Eigensolver (VQE) algorithm.
        
        Args:
            hamiltonian: The Hamiltonian matrix whose ground state energy we want to find
            ansatz: Function that applies the ansatz circuit with given parameters
            initial_params: Initial parameters for the ansatz
            optimizer: Classical optimization function
            n_qubits: Number of qubits in the system
            
        Returns:
            Dictionary with optimized parameters and energy
        """
        # Define the objective function (energy)
        def objective(params: List[float]) -> float:
            # Create a circuit with the ansatz
            circuit = QuantumCircuit(n_qubits)
            ansatz(circuit, params)
            
            # Get the state vector
            statevector = circuit.get_statevector()
            
            # Calculate the expectation value ⟨ψ|H|ψ⟩
            energy = np.real(np.vdot(statevector, hamiltonian @ statevector))
            
            return energy
            
        # Run the classical optimizer
        optimal_params, minimal_energy = optimizer(objective, initial_params)
        
        # Create the optimal circuit
        optimal_circuit = QuantumCircuit(n_qubits)
        ansatz(optimal_circuit, optimal_params)
        
        # Get the final state
        optimal_state = optimal_circuit.get_statevector()
        
        return {
            "optimal_params": optimal_params,
            "ground_state_energy": minimal_energy,
            "ground_state": optimal_state
        }

# ============================================================================
# Membrane Module
# ============================================================================

class Membrane:
    """Base class for membrane filtering and validation."""
    
    def __init__(self):
        """Initialize a membrane with default configuration."""
        self.filters = []
        self.transformers = []
        self.validators = []
        self.cache = {}
        self.stats = {"processed": 0, "filtered": 0, "validated": 0, "transformed": 0}
        
    def add_filter(self, filter_func: Callable[[Any], bool], name: str = None):
        """Add a filter function to the membrane."""
        filter_name = name or f"filter_{len(self.filters)}"
        self.filters.append((filter_func, filter_name))
        return self
        
    def add_transformer(self, transformer_func: Callable[[Any], Any], name: str = None):
        """Add a transformer function to the membrane."""
        transformer_name = name or f"transformer_{len(self.transformers)}"
        self.transformers.append((transformer_func, transformer_name))
        return self
        
    def add_validator(self, validator_func: Callable[[Any], Tuple[bool, str]], name: str = None):
        """Add a validator function to the membrane."""
        validator_name = name or f"validator_{len(self.validators)}"
        self.validators.append((validator_func, validator_name))
        return self
        
    def process(self, data: Any) -> Tuple[bool, Any, Dict[str, Any]]:
        """
        Process data through the membrane.
        
        Returns:
            Tuple of (success, processed_data, metadata)
        """
        self.stats["processed"] += 1
        metadata = {"original_type": type(data).__name__, "filters": [], "transformers": [], "validators": []}
        
        # Check cache first
        cache_key = self._get_cache_key(data)
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        # Apply filters
        for filter_func, filter_name in self.filters:
            if not filter_func(data):
                metadata["filters"].append({"name": filter_name, "passed": False})
                self.stats["filtered"] += 1
                
                result = (False, None, metadata)
                self.cache[cache_key] = result
                return result
            metadata["filters"].append({"name": filter_name, "passed": True})
            
        # Apply transformers
        transformed_data = data
        for transformer_func, transformer_name in self.transformers:
            try:
                transformed_data = transformer_func(transformed_data)
                metadata["transformers"].append({
                    "name": transformer_name, 
                    "success": True,
                    "output_type": type(transformed_data).__name__
                })
                self.stats["transformed"] += 1
            except Exception as e:
                metadata["transformers"].append({
                    "name": transformer_name, 
                    "success": False,
                    "error": str(e)
                })
                
                result = (False, None, metadata)
                self.cache[cache_key] = result
                return result
                
        # Apply validators
        for validator_func, validator_name in self.validators:
            try:
                valid, message = validator_func(transformed_data)
                metadata["validators"].append({
                    "name": validator_name,
                    "valid": valid,
                    "message": message
                })
                
                if not valid:
                    self.stats["validated"] += 1
                    
                    result = (False, None, metadata)
                    self.cache[cache_key] = result
                    return result
            except Exception as e:
                metadata["validators"].append({
                    "name": validator_name,
                    "valid": False,
                    "error": str(e)
                })
                
                result = (False, None, metadata)
                self.cache[cache_key] = result
                return result
                
        # All checks passed
        result = (True, transformed_data, metadata)
        self.cache[cache_key] = result
        return result
        
    def _get_cache_key(self, data: Any) -> str:
        """Generate a cache key for the data."""
        try:
            # Try using the data directly as a key
            if isinstance(data, (str, int, float, bool)):
                return f"{type(data).__name__}:{str(data)}"
                
            # For more complex types, use a hash
            if hasattr(data, '__hash__') and callable(data.__hash__):
                return f"hash:{hash(data)}"
                
            # For unhashable types, use a representation
            return f"repr:{repr(data)[:100]}"
        except:
            # Fall back to object ID
            return f"id:{id(data)}"
            
    def clear_cache(self):
        """Clear the membrane's cache."""
        self.cache = {}
        
    def reset_stats(self):
        """Reset the membrane's statistics."""
        self.stats = {"processed": 0, "filtered": 0, "validated": 0, "transformed": 0}
        
    def get_stats(self) -> Dict[str, Any]:
        """Get the membrane's statistics."""
        return self.stats.copy()

class DataMembrane(Membrane):
    """Specialized membrane for data processing."""
    
    def __init__(self, expected_schema: Optional[Dict[str, Any]] = None):
        """Initialize a data membrane with an optional expected schema."""
        super().__init__()
        self.expected_schema = expected_schema
        
        # Add default validators if schema is provided
        if expected_schema:
            self.add_validator(self._schema_validator, "schema_validator")
            
    def _schema_validator(self, data: Any) -> Tuple[bool, str]:
        """Validate data against the expected schema."""
        if self.expected_schema is None:
            return True, "No schema to validate against"
            
        if not isinstance(data, dict):
            return False, f"Expected dictionary, got {type(data).__name__}"
            
        # Check required fields
        for field, field_spec in self.expected_schema.items():
            if field_spec.get("required", False) and field not in data:
                return False, f"Missing required field: {field}"
                
            if field in data:
                # Check type
                expected_type = field_spec.get("type")
                if expected_type and not isinstance(data[field], expected_type):
                    return False, f"Field {field} has incorrect type: {type(data[field]).__name__}, expected {expected_type.__name__}"
                    
                # Check constraints
                constraints = field_spec.get("constraints", {})
                
                # Numeric constraints
                if "min" in constraints and data[field] < constraints["min"]:
                    return False, f"Field {field} value {data[field]} is less than minimum {constraints['min']}"
                if "max" in constraints and data[field] > constraints["max"]:
                    return False, f"Field {field} value {data[field]} is greater than maximum {constraints['max']}"
                    
                # String constraints
                if "pattern" in constraints and not re.match(constraints["pattern"], data[field]):
                    return False, f"Field {field} value {data[field]} does not match pattern {constraints['pattern']}"
                if "max_length" in constraints and len(data[field]) > constraints["max_length"]:
                    return False, f"Field {field} length {len(data[field])} exceeds maximum {constraints['max_length']}"
                if "min_length" in constraints and len(data[field]) < constraints["min_length"]:
                    return False, f"Field {field} length {len(data[field])} is less than minimum {constraints['min_length']}"
                    
                # Enum constraints
                if "enum" in constraints and data[field] not in constraints["enum"]:
                    return False, f"Field {field} value {data[field]} is not in allowed values: {constraints['enum']}"
                    
        return True, "Data validated successfully"
        
    def add_type_filter(self, expected_type: type):
        """Add a filter that checks for a specific type."""
        return self.add_filter(
            lambda data: isinstance(data, expected_type),
            f"type_filter_{expected_type.__name__}"
        )
        
    def add_null_filter(self):
        """Add a filter that rejects null/None values."""
        return self.add_filter(
            lambda data: data is not None,
            "null_filter"
        )
        
    def add_empty_filter(self):
        """Add a filter that rejects empty collections."""
        return self.add_filter(
            lambda data: not (hasattr(data, "__len__") and len(data) == 0),
            "empty_filter"
        )
        
    def add_range_validator(self, min_val: float, max_val: float):
        """Add a validator that checks if numeric values are within a range."""
        def range_validator(data: Any) -> Tuple[bool, str]:
            if not isinstance(data, (int, float)):
                return False, f"Expected numeric type, got {type(data).__name__}"
            if data < min_val:
                return False, f"Value {data} is less than minimum {min_val}"
            if data > max_val:
                return False, f"Value {data} is greater than maximum {max_val}"
            return True, "Value is within range"
            
        return self.add_validator(range_validator, f"range_validator_{min_val}_{max_val}")
        
    def add_length_validator(self, min_length: int, max_length: int):
        """Add a validator that checks if collections have proper length."""
        def length_validator(data: Any) -> Tuple[bool, str]:
            if not hasattr(data, "__len__"):
                return False, f"Object of type {type(data).__name__} has no length"
            length = len(data)
            if length < min_length:
                return False, f"Length {length} is less than minimum {min_length}"
            if length > max_length:
                return False, f"Length {length} exceeds maximum {max_length}"
            return True, "Length is within range"
            
        return self.add_validator(length_validator, f"length_validator_{min_length}_{max_length}")
        
    def add_regex_validator(self, pattern: str):
        """Add a validator that checks if strings match a regex pattern."""
        import re
        compiled_pattern = re.compile(pattern)
        
        def regex_validator(data: Any) -> Tuple[bool, str]:
            if not isinstance(data, str):
                return False, f"Expected string, got {type(data).__name__}"
            if not compiled_pattern.match(data):
                return False, f"String '{data}' does not match pattern '{pattern}'"
            return True, "String matches pattern"
            
        return self.add_validator(regex_validator, f"regex_validator_{pattern}")
        
    def add_json_transformer(self):
        """Add a transformer that parses JSON strings."""
        import json
        
        def json_transformer(data: Any) -> Any:
            if isinstance(data, str):
                return json.loads(data)
            return data
            
        return self.add_transformer(json_transformer, "json_transformer")
        
    def add_string_transformer(self):
        """Add a transformer that converts data to strings."""
        return self.add_transformer(
            lambda data: str(data),
            "string_transformer"
        )
        
    def add_numpy_transformer(self):
        """Add a transformer that converts lists to numpy arrays."""
        def numpy_transformer(data: Any) -> Any:
            if isinstance(data, (list, tuple)) and all(isinstance(x, (int, float)) for x in data):
                return np.array(data)
            return data
            
        return self.add_transformer(numpy_transformer, "numpy_transformer")

class BinaryMembrane(Membrane):
    """Specialized membrane for binary data processing."""
    
    def __init__(self, expected_magic_numbers: Optional[List[bytes]] = None):
        """Initialize a binary membrane with optional expected magic numbers."""
        super().__init__()
        self.expected_magic_numbers = expected_magic_numbers
        
        # Add default validators if magic numbers are provided
        if expected_magic_numbers:
            self.add_validator(self._magic_number_validator, "magic_number_validator")
            
    def _magic_number_validator(self, data: bytes) -> Tuple[bool, str]:
        """Validate binary data against expected magic numbers."""
        if not self.expected_magic_numbers:
            return True, "No magic numbers to validate against"
            
        if not isinstance(data, bytes):
            return False, f"Expected bytes, got {type(data).__name__}"
            
        if not data:
            return False, "Empty data"
            
        # Check if data starts with any of the expected magic numbers
        for magic in self.expected_magic_numbers:
            if data.startswith(magic):
                return True, f"Magic number matched: {magic.hex()}"
                
        return False, f"No matching magic number found in {data[:20].hex()}"
        
    def add_size_filter(self, min_size: int, max_size: Optional[int] = None):
        """Add a filter that checks if binary data size is within range."""
        def size_filter(data: Any) -> bool:
            if not isinstance(data, bytes):
                return False
            if len(data) < min_size:
                return False
            if max_size is not None and len(data) > max_size:
                return False
            return True
            
        return self.add_filter(size_filter, f"size_filter_{min_size}_{max_size}")
        
    def add_checksum_validator(self, checksum_func: Callable[[bytes], bytes], expected_checksum: bytes):
        """Add a validator that checks if data matches an expected checksum."""
        def checksum_validator(data: bytes) -> Tuple[bool, str]:
            if not isinstance(data, bytes):
                return False, f"Expected bytes, got {type(data).__name__}"
                
            computed_checksum = checksum_func(data)
            if computed_checksum != expected_checksum:
                return False, f"Checksum mismatch: got {computed_checksum.hex()}, expected {expected_checksum.hex()}"
                
            return True, "Checksum validated"
            
        return self.add_validator(checksum_validator, "checksum_validator")
        
    def add_decompression_transformer(self, algorithm: str = 'zlib'):
        """Add a transformer that decompresses binary data."""
        import zlib
        
        def decompress_transformer(data: bytes) -> bytes:
            if not isinstance(data, bytes):
                return data
                
            if algorithm == 'zlib':
                return zlib.decompress(data)
            elif algorithm == 'gzip':
                return zlib.decompress(data, 16 + zlib.MAX_WBITS)
            else:
                raise ValueError(f"Unsupported compression algorithm: {algorithm}")
                
        return self.add_transformer(decompress_transformer, f"decompress_{algorithm}_transformer")
        
    def add_encryption_transformer(self, key: bytes, algorithm: str = 'xor'):
        """Add a transformer that decrypts binary data."""
        def decrypt_xor(data: bytes, key: bytes) -> bytes:
            """Simple XOR decryption."""
            result = bytearray(len(data))
            for i, b in enumerate(data):
                result[i] = b ^ key[i % len(key)]
            return bytes(result)
            
        def decrypt_transformer(data: bytes) -> bytes:
            if not isinstance(data, bytes):
                return data
                
            if algorithm == 'xor':
                return decrypt_xor(data, key)
            else:
                raise ValueError(f"Unsupported encryption algorithm: {algorithm}")
                
        return self.add_transformer(decrypt_transformer, f"decrypt_{algorithm}_transformer")
        
    def add_hex_transformer(self):
        """Add a transformer that converts binary data to hex strings."""
        return self.add_transformer(
            lambda data: data.hex() if isinstance(data, bytes) else data,
            "hex_transformer"
        )
        
    def add_hash_transformer(self, algorithm: str = 'sha256'):
        """Add a transformer that hashes binary data."""
        def hash_transformer(data: Any) -> str:
            if isinstance(data, bytes):
                if algorithm == 'md5':
                    return hashlib.md5(data).hexdigest()
                elif algorithm == 'sha1':
                    return hashlib.sha1(data).hexdigest()
                elif algorithm == 'sha256':
                    return hashlib.sha256(data).hexdigest()
                else:
                    raise ValueError(f"Unsupported hash algorithm: {algorithm}")
            return data
            
        return self.add_transformer(hash_transformer, f"hash_{algorithm}_transformer")

# ============================================================================
# Node Network Module
# ============================================================================

class NodeType(Enum):
    """Types of nodes in the network."""
    INPUT = auto()
    PROCESSING = auto()
    OUTPUT = auto()
    SUPERNODE = auto()
    STORAGE = auto()
    FILTERING = auto()
    MEMORY = auto()
    VISUALIZATION = auto()
    QUANTUM = auto()
    CUSTOM = auto()

class MessageType(Enum):
    """Types of messages that can be passed between nodes."""
    DATA = auto()
    COMMAND = auto()
    STATUS = auto()
    ERROR = auto()
    INFO = auto()
    RESULT = auto()
    REQUEST = auto()
    RESPONSE = auto()
    EVENT = auto()
    CUSTOM = auto()

@dataclass
class Message:
    """Message passed between nodes."""
    msg_type: MessageType
    content: Any
    sender_id: str
    timestamp: float = field(default_factory=time.time)
    message_id: str = field(default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest())
    priority: int = 0
    ttl: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class Node(ABC):
    """Base class for all nodes in the network."""
    
    def __init__(self, node_id: str, node_type: NodeType):
        """Initialize a node with an ID and type."""
        self.node_id = node_id
        self.node_type = node_type
        self.connections = {}  # Map of node_id -> Node
        self.message_queue = []
        self.processed_messages = {}
        self.state = {}
        self.active = True
        self.membrane = Membrane()  # Default membrane for filtering
        
    def connect(self, node: 'Node', bidirectional: bool = True) -> 'Node':
        """Connect this node to another node."""
        self.connections[node.node_id] = node
        if bidirectional:
            node.connections[self.node_id] = self
        return self
        
    def disconnect(self, node_id: str, bidirectional: bool = True) -> 'Node':
        """Disconnect from another node."""
        if node_id in self.connections:
            node = self.connections[node_id]
            del self.connections[node_id]
            
            if bidirectional and self.node_id in node.connections:
                del node.connections[self.node_id]
                
        return self
        
    def send(self, message: Message, target_id: str) -> bool:
        """Send a message to a specific node."""
        if target_id not in self.connections:
            logger.warning(f"Node {self.node_id} cannot send to unknown node {target_id}")
            return False
            
        target = self.connections[target_id]
        return target.receive(message)
        
    def broadcast(self, message: Message) -> int:
        """Broadcast a message to all connected nodes."""
        success_count = 0
        
        for node_id in self.connections:
            if self.send(message, node_id):
                success_count += 1
                
        return success_count
        
    def receive(self, message: Message) -> bool:
        """Receive a message and add it to the queue."""
        if not self.active:
            logger.warning(f"Node {self.node_id} is inactive, rejecting message {message.message_id}")
            return False
            
        # Check if we've already processed this message
        if message.message_id in self.processed_messages:
            logger.warning(f"Node {self.node_id} already processed message {message.message_id}")
            return False
            
        # Filter message through membrane
        success, processed_message, metadata = self.membrane.process(message)
        
        if not success:
            logger.warning(f"Node {self.node_id} membrane rejected message {message.message_id}: {metadata}")
            return False
            
        # Add to queue based on priority
        heapq.heappush(self.message_queue, (-message.priority, message))
        return True
        
    def process_next(self) -> Optional[Message]:
        """Process the next message in the queue."""
        if not self.message_queue:
            return None
            
        _, message = heapq.heappop(self.message_queue)
        
        # Mark as processed
        self.processed_messages[message.message_id] = time.time()
        
        # Process the message
        self._process_message(message)
        
        return message
        
    @abstractmethod
    def _process_message(self, message: Message) -> Any:
        """Process a message (to be implemented by subclasses)."""
        pass
        
    def process_all(self) -> int:
        """Process all messages in the queue."""
        count = 0
        while self.message_queue:
            self.process_next()
            count += 1
        return count
        
    def create_message(self, msg_type: MessageType, content: Any, priority: int = 0) -> Message:
        """Create a new message originating from this node."""
        return Message(
            msg_type=msg_type,
            content=content,
            sender_id=self.node_id,
            priority=priority
        )
        
    def activate(self):
        """Activate the node."""
        self.active = True
        
    def deactivate(self):
        """Deactivate the node."""
        self.active = False
        
    def save_state(self, filepath: str):
        """Save the node's state to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'node_id': self.node_id,
                'node_type': self.node_type,
                'state': self.state,
                'active': self.active
            }, f)
            
    def load_state(self, filepath: str):
        """Load the node's state from a file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        if data['node_id'] != self.node_id:
            logger.warning(f"Loading state from node {data['node_id']} into node {self.node_id}")
            
        self.state = data['state']
        self.active = data['active']
        
    def get_connections(self) -> List[str]:
        """Get a list of all connected node IDs."""
        return list(self.connections.keys())
        
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about this node."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.name,
            "connections": len(self.connections),
            "queue_size": len(self.message_queue),
            "processed_messages": len(self.processed_messages),
            "active": self.active,
            "membrane_stats": self.membrane.get_stats()
        }

class ProcessingNode(Node):
    """Node that processes data with a specific function."""
    
    def __init__(self, node_id: str, process_func: Callable[[Any], Any]):
        """Initialize a processing node with a processing function."""
        super().__init__(node_id, NodeType.PROCESSING)
        self.process_func = process_func
        self.results = {}
        
    def _process_message(self, message: Message) -> Any:
        """Process a message using the processing function."""
        if message.msg_type == MessageType.DATA:
            try:
                # Process the data
                result = self.process_func(message.content)
                
                # Store the result
                self.results[message.message_id] = result
                
                # Create a result message
                result_message = self.create_message(
                    MessageType.RESULT,
                    result,
                    priority=message.priority
                )
                
                # Set metadata
                result_message.metadata = {
                    "original_message_id": message.message_id,
                    "processing_time": time.time() - message.timestamp,
                    "success": True
                }
                
                # Broadcast the result
                self.broadcast(result_message)
                
                return result
            except Exception as e:
                # Create an error message
                error_message = self.create_message(
                    MessageType.ERROR,
                    str(e),
                    priority=message.priority
                )
                
                # Set metadata
                error_message.metadata = {
                    "original_message_id": message.message_id,
                    "error_type": type(e).__name__,
                    "stack_trace": traceback.format_exc()
                }
                
                # Broadcast the error
                self.broadcast(error_message)
                
                return None
        else:
            # For non-data messages, just log and ignore
            logger.info(f"ProcessingNode {self.node_id} received non-data message: {message.msg_type}")
            return None
            
    def get_result(self, message_id: str) -> Optional[Any]:
        """Get the result of processing a specific message."""
        return self.results.get(message_id)
        
    def clear_results(self):
        """Clear all stored results."""
        self.results = {}

class SuperNode(Node):
    """A node that contains and manages other nodes."""
    
    def __init__(self, node_id: str):
        """Initialize a super node."""
        super().__init__(node_id, NodeType.SUPERNODE)
        self.subnodes = {}  # Map of node_id -> Node
        self.routing_table = {}  # Map of message types to target node IDs
        
    def add_node(self, node: Node) -> 'SuperNode':
        """Add a node to this super node."""
        self.subnodes[node.node_id] = node
        return self
        
    def remove_node(self, node_id: str) -> Optional[Node]:
        """Remove a node from this super node."""
        if node_id in self.subnodes:
            node = self.subnodes[node_id]
            del self.subnodes[node_id]
            return node
        return None
        
    def add_route(self, msg_type: MessageType, target_id: str) -> 'SuperNode':
        """Add a routing rule for a message type."""
        if target_id not in self.subnodes and target_id not in self.connections:
            logger.warning(f"SuperNode {self.node_id} adding route to unknown node {target_id}")
            
        if msg_type not in self.routing_table:
            self.routing_table[msg_type] = []
            
        self.routing_table[msg_type].append(target_id)
        return self
        
    def _process_message(self, message: Message) -> Any:
        """Process a message by routing it to the appropriate subnode."""
        if message.msg_type in self.routing_table:
            # Route to all appropriate targets
            targets = self.routing_table[message.msg_type]
            results = []
            
            for target_id in targets:
                if target_id in self.subnodes:
                    # Internal routing
                    success = self.subnodes[target_id].receive(message)
                    if success:
                        # Process immediately
                        result = self.subnodes[target_id].process_next()
                        results.append(result)
                elif target_id in self.connections:
                    # External routing
                    success = self.send(message, target_id)
                    if success:
                        results.append(True)
                else:
                    logger.warning(f"SuperNode {self.node_id} cannot route message to unknown node {target_id}")
                    
            return results
        else:
            # Process in all subnodes
            for node in self.subnodes.values():
                node.receive(message)
                node.process_next()
                
            # Also broadcast to external connections
            self.broadcast(message)
            
            return None
            
    def process_all_subnodes(self) -> Dict[str, int]:
        """Process all messages in all subnodes."""
        results = {}
        
        for node_id, node in self.subnodes.items():
            count = node.process_all()
            results[node_id] = count
            
        return results
        
    def activate_all(self):
        """Activate all subnodes."""
        for node in self.subnodes.values():
            node.activate()
            
    def deactivate_all(self):
        """Deactivate all subnodes."""
        for node in self.subnodes.values():
            node.deactivate()
            
    def get_subnode(self, node_id: str) -> Optional[Node]:
        """Get a specific subnode by ID."""
        return self.subnodes.get(node_id)
        
    def get_all_subnodes(self) -> Dict[str, Node]:
        """Get all subnodes."""
        return self.subnodes.copy()
        
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about this supernode and its subnodes."""
        stats = super().get_stats()
        
        # Add supernode-specific stats
        stats["subnodes"] = len(self.subnodes)
        stats["routes"] = {msg_type.name: targets for msg_type, targets in self.routing_table.items()}
        
        # Add subnode stats
        subnode_stats = {}
        for node_id, node in self.subnodes.items():
            subnode_stats[node_id] = node.get_stats()
            
        stats["subnode_stats"] = subnode_stats
        
        return stats

class InputNode(Node):
    """Node that serves as an input point for data."""
    
    def __init__(self, node_id: str, input_type: type):
        """Initialize an input node with an expected input type."""
        super().__init__(node_id, NodeType.INPUT)
        self.input_type = input_type
        
        # Add a type filter to the membrane
        self.membrane.add_filter(
            lambda data: isinstance(data.content, self.input_type),
            f"input_type_filter_{self.input_type.__name__}"
        )
        
    def _process_message(self, message: Message) -> Any:
        """Process input messages and forward them."""
        # Simply forward all messages to all connected nodes
        self.broadcast(message)
        return message.content
        
    def input_data(self, data: Any, priority: int = 0) -> bool:
        """Input data to the node and broadcast it."""
        if not isinstance(data, self.input_type):
            logger.warning(f"InputNode {self.node_id} expected {self.input_type.__name__}, got {type(data).__name__}")
            return False
            
        # Create a data message
        message = self.create_message(MessageType.DATA, data, priority)
        
        # Add to the queue and process immediately
        heapq.heappush(self.message_queue, (-priority, message))
        self.process_next()
        
        return True

class OutputNode(Node):
    """Node that serves as an output point for results."""
    
    def __init__(self, node_id: str, output_handler: Callable[[Any], None] = None):
        """Initialize an output node with an optional output handler."""
        super().__init__(node_id, NodeType.OUTPUT)
        self.output_handler = output_handler or (lambda x: None)
        self.outputs = []
        
    def _process_message(self, message: Message) -> Any:
        """Process output messages."""
        # Store the output
        self.outputs.append((message, time.time()))
        
        # Call the output handler
        self.output_handler(message.content)
        
        return message.content
        
    def get_outputs(self, n: Optional[int] = None) -> List[Tuple[Message, float]]:
        """Get the latest n outputs with timestamps."""
        if n is None:
            return self.outputs.copy()
        return self.outputs[-n:]
        
    def clear_outputs(self):
        """Clear all stored outputs."""
        self.outputs = []
        
    def set_output_handler(self, handler: Callable[[Any], None]):
        """Set a new output handler."""
        self.output_handler = handler

class StorageNode(Node):
    """Node that stores data for later retrieval."""
    
    def __init__(self, node_id: str, capacity: Optional[int] = None):
        """Initialize a storage node with an optional capacity limit."""
        super().__init__(node_id, NodeType.STORAGE)
        self.capacity = capacity
        self.storage = {}  # Key-value store
        self.timestamps = {}  # Timestamp for each key
        self.access_counts = {}  # Access count for each key
        
    def _process_message(self, message: Message) -> Any:
        """Process storage-related messages."""
        if message.msg_type == MessageType.DATA:
            # Check if the message has a key in its metadata
            key = message.metadata.get("storage_key")
            
            if key is None:
                # Generate a key based on the message ID
                key = f"msg_{message.message_id}"
                
            # Store the data
            success = self.store(key, message.content)
            
            # Create a status message
            status_message = self.create_message(
                MessageType.STATUS,
                {"success": success, "key": key},
                priority=message.priority
            )
            
            # Broadcast the status
            self.broadcast(status_message)
            
            return success
        elif message.msg_type == MessageType.REQUEST:
            # Handle data retrieval requests
            request = message.content
            
            if isinstance(request, dict) and "key" in request:
                key = request["key"]
                data = self.retrieve(key)
                
                # Create a response message
                response_message = self.create_message(
                    MessageType.RESPONSE,
                    {"key": key, "data": data},
                    priority=message.priority
                )
                
                # Send the response back to the sender
                self.send(response_message, message.sender_id)
                
                return data
                
        return None
        
    def store(self, key: str, data: Any) -> bool:
        """Store data under a key."""
        # Check capacity
        if self.capacity is not None and len(self.storage) >= self.capacity and key not in self.storage:
            # Implement LRU eviction
            if self.timestamps:
                # Find the least recently used key
                lru_key = min(self.timestamps, key=self.timestamps.get)
                # Remove it
                del self.storage[lru_key]
                del self.timestamps[lru_key]
                del self.access_counts[lru_key]
                
        # Store the data
        self.storage[key] = data
        self.timestamps[key] = time.time()
        self.access_counts[key] = self.access_counts.get(key, 0)
        
        return True
        
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve data by key."""
        if key in self.storage:
            # Update timestamp and access count
            self.timestamps[key] = time.time()
            self.access_counts[key] = self.access_counts.get(key, 0) + 1
            
            return self.storage[key]
            
        return None
        
    def delete(self, key: str) -> bool:
        """Delete data by key."""
        if key in self.storage:
            del self.storage[key]
            del self.timestamps[key]
            del self.access_counts[key]
            return True
            
        return False
        
    def clear(self):
        """Clear all stored data."""
        self.storage = {}
        self.timestamps = {}
        self.access_counts = {}
        
    def list_keys(self) -> List[str]:
        """List all stored keys."""
        return list(self.storage.keys())
        
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        stats = super().get_stats()
        
        # Add storage-specific stats
        stats["storage_size"] = len(self.storage)
        stats["storage_capacity"] = self.capacity
        
        if self.storage:
            stats["oldest_key"] = min(self.timestamps, key=self.timestamps.get)
            stats["newest_key"] = max(self.timestamps, key=self.timestamps.get)
            stats["most_accessed_key"] = max(self.access_counts, key=self.access_counts.get)
            stats["most_access_count"] = max(self.access_counts.values()) if self.access_counts else 0
            
        return stats

class FilteringNode(Node):
    """Node that filters messages based on criteria."""
    
    def __init__(self, node_id: str, filter_func: Callable[[Message], bool]):
        """Initialize a filtering node with a filter function."""
        super().__init__(node_id, NodeType.FILTERING)
        self.filter_func = filter_func
        self.filtered_count = 0
        self.passed_count = 0
        
    def _process_message(self, message: Message) -> Any:
        """Process messages by filtering them."""
        # Apply the filter
        if self.filter_func(message):
            # Message passed the filter
            self.passed_count += 1
            
            # Forward to all connected nodes
            self.broadcast(message)
            
            return True
        else:
            # Message didn't pass the filter
            self.filtered_count += 1
            
            # Create a status message
            status_message = self.create_message(
                MessageType.STATUS,
                {"filtered": True, "message_id": message.message_id},
                priority=message.priority
            )
            
            # Send status message back to sender
            self.send(status_message, message.sender_id)
            
            return False
            
    def get_stats(self) -> Dict[str, Any]:
        """Get filtering statistics."""
        stats = super().get_stats()
        
        # Add filtering-specific stats
        stats["filtered_count"] = self.filtered_count
        stats["passed_count"] = self.passed_count
        stats["pass_rate"] = self.passed_count / (self.passed_count + self.filtered_count) if (self.passed_count + self.filtered_count) > 0 else 0
        
        return stats

class VisualizationNode(Node):
    """Node that creates visualizations of data."""
    
    def __init__(self, node_id: str, visualize_func: Callable[[Any], Any]):
        """Initialize a visualization node with a visualization function."""
        super().__init__(node_id, NodeType.VISUALIZATION)
        self.visualize_func = visualize_func
        self.visualizations = {}
        
    def _process_message(self, message: Message) -> Any:
        """Process messages by creating visualizations."""
        if message.msg_type == MessageType.DATA:
            try:
                # Create visualization
                visualization = self.visualize_func(message.content)
                
                # Store the visualization
                self.visualizations[message.message_id] = visualization
                
                # Create a result message
                result_message = self.create_message(
                    MessageType.RESULT,
                    visualization,
                    priority=message.priority
                )
                
                # Set metadata
                result_message.metadata = {
                    "original_message_id": message.message_id,
                    "visualization_type": type(visualization).__name__
                }
                
                # Broadcast the result
                self.broadcast(result_message)
                
                return visualization
            except Exception as e:
                # Create an error message
                error_message = self.create_message(
                    MessageType.ERROR,
                    str(e),
                    priority=message.priority
                )
                
                # Set metadata
                error_message.metadata = {
                    "original_message_id": message.message_id,
                    "error_type": type(e).__name__,
                    "stack_trace": traceback.format_exc()
                }
                
                # Broadcast the error
                self.broadcast(error_message)
                
                return None
                
        return None
        
    def get_visualization(self, message_id: str) -> Optional[Any]:
        """Get a visualization by message ID."""
        return self.visualizations.get(message_id)
        
    def clear_visualizations(self):
        """Clear all stored visualizations."""
        self.visualizations = {}

class QuantumNode(Node):
    """Node that performs quantum operations."""
    
    def __init__(self, node_id: str, simulator: QuantumSimulator):
        """Initialize a quantum node with a quantum simulator."""
        super().__init__(node_id, NodeType.QUANTUM)
        self.simulator = simulator
        self.circuits = {}
        self.results = {}
        
    def _process_message(self, message: Message) -> Any:
        """Process messages with quantum operations."""
        if message.msg_type == MessageType.COMMAND:
            command = message.content
            
            if not isinstance(command, dict):
                logger.warning(f"QuantumNode {self.node_id} expected dict command, got {type(command).__name__}")
                return None
                
            # Handle different quantum commands
            if "operation" in command:
                operation = command["operation"]
                
                if operation == "create_circuit":
                    # Create a new quantum circuit
                    n_qubits = command.get("n_qubits", 1)
                    circuit_id = command.get("circuit_id", f"circuit_{message.message_id}")
                    
                    circuit = self.simulator.create_circuit(n_qubits)
                    self.circuits[circuit_id] = circuit
                    
                    # Create a status message
                    status_message = self.create_message(
                        MessageType.STATUS,
                        {"circuit_created": True, "circuit_id": circuit_id, "n_qubits": n_qubits},
                        priority=message.priority
                    )
                    
                    # Broadcast the status
                    self.broadcast(status_message)
                    
                    return circuit
                    
                elif operation == "add_gate":
                    # Add a gate to a circuit
                    circuit_id = command.get("circuit_id")
                    gate = command.get("gate")
                    targets = command.get("targets", [])
                    params = command.get("params")
                    
                    if circuit_id not in self.circuits:
                        logger.warning(f"QuantumNode {self.node_id} unknown circuit {circuit_id}")
                        return None
                        
                    circuit = self.circuits[circuit_id]
                    circuit.add_gate(gate, targets, params)
                    
                    return True
                    
                elif operation == "run_circuit":
                    # Run a quantum circuit
                    circuit_id = command.get("circuit_id")
                    shots = command.get("shots", 1024)
                    
                    if circuit_id not in self.circuits:
                        logger.warning(f"QuantumNode {self.node_id} unknown circuit {circuit_id}")
                        return None
                        
                    circuit = self.circuits[circuit_id]
                    results = self.simulator.run_circuit(circuit, shots)
                    
                    # Store the results
                    result_id = command.get("result_id", f"result_{message.message_id}")
                    self.results[result_id] = results
                    
                    # Create a result message
                    result_message = self.create_message(
                        MessageType.RESULT,
                        {"result_id": result_id, "results": results},
                        priority=message.priority
                    )
                    
                    # Broadcast the result
                    self.broadcast(result_message)
                    
                    return results
                    
                elif operation == "qft":
                    # Create a Quantum Fourier Transform circuit
                    n_qubits = command.get("n_qubits", 3)
                    circuit_id = command.get("circuit_id", f"qft_{message.message_id}")
                    
                    circuit = self.simulator.quantum_fourier_transform(n_qubits)
                    self.circuits[circuit_id] = circuit
                    
                    # Create a status message
                    status_message = self.create_message(
                        MessageType.STATUS,
                        {"qft_created": True, "circuit_id": circuit_id, "n_qubits": n_qubits},
                        priority=message.priority
                    )
                    
                    # Broadcast the status
                    self.broadcast(status_message)
                    
                    return circuit
                    
                elif operation == "grover":
                    # Run Grover's algorithm
                    n_qubits = command.get("n_qubits", 3)
                    marked_states = command.get("marked_states", [])
                    iterations = command.get("iterations")
                    circuit_id = command.get("circuit_id", f"grover_{message.message_id}")
                    
                    # Create oracle function
                    def oracle_function(state: int) -> bool:
                        return state in marked_states
                        
                    circuit = self.simulator.simulate_grover(n_qubits, oracle_function, iterations)
                    self.circuits[circuit_id] = circuit
                    
                    # Run the circuit
                    shots = command.get("shots", 1024)
                    results = self.simulator.run_circuit(circuit, shots)
                    
                    # Store the results
                    result_id = command.get("result_id", f"result_{message.message_id}")
                    self.results[result_id] = results
                    
                    # Create a result message
                    result_message = self.create_message(
                        MessageType.RESULT,
                        {"result_id": result_id, "results": results, "circuit_id": circuit_id},
                        priority=message.priority
                    )
                    
                    # Broadcast the result
                    self.broadcast(result_message)
                    
                    return results
                    
                elif operation == "phase_estimation":
                    # Run Quantum Phase Estimation
                    n_counting_qubits = command.get("n_counting_qubits", 3)
                    unitary_matrix = command.get("unitary_matrix")
                    circuit_id = command.get("circuit_id", f"qpe_{message.message_id}")
                    
                    if unitary_matrix is None:
                        logger.warning(f"QuantumNode {self.node_id} missing unitary_matrix for phase_estimation")
                        return None
                        
                    # Convert to numpy array if needed
                    if not isinstance(unitary_matrix, np.ndarray):
                        unitary_matrix = np.array(unitary_matrix, dtype=np.complex128)
                        
                    circuit = self.simulator.simulate_quantum_phase_estimation(n_counting_qubits, unitary_matrix)
                    self.circuits[circuit_id] = circuit
                    
                    # Run the circuit
                    shots = command.get("shots", 1024)
                    results = self.simulator.run_circuit(circuit, shots)
                    
                    # Store the results
                    result_id = command.get("result_id", f"result_{message.message_id}")
                    self.results[result_id] = results
                    
                    # Create a result message
                    result_message = self.create_message(
                        MessageType.RESULT,
                        {"result_id": result_id, "results": results, "circuit_id": circuit_id},
                        priority=message.priority
                    )
                    
                    # Broadcast the result
                    self.broadcast(result_message)
                    
                    return results
                    
        return None
        
    def get_circuit(self, circuit_id: str) -> Optional[QuantumCircuit]:
        """Get a quantum circuit by ID."""
        return self.circuits.get(circuit_id)
        
    def get_result(self, result_id: str) -> Optional[Dict[str, int]]:
        """Get a quantum result by ID."""
        return self.results.get(result_id)
        
    def clear_circuits(self):
        """Clear all stored circuits."""
        self.circuits = {}
        
    def clear_results(self):
        """Clear all stored results."""
        self.results = {}

class NetworkManager:
    """Manager for building and controlling node networks."""
    
    def __init__(self):
        """Initialize a network manager."""
        self.nodes = {}  # Map of node_id -> Node
        self.node_groups = {}  # Map of group_name -> List[node_id]
        
    def add_node(self, node: Node) -> 'NetworkManager':
        """Add a node to the network."""
        if node.node_id in self.nodes:
            logger.warning(f"NetworkManager replacing existing node {node.node_id}")
            
        self.nodes[node.node_id] = node
        return self
        
    def remove_node(self, node_id: str) -> Optional[Node]:
        """Remove a node from the network."""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            
            # Disconnect from all other nodes
            for other_id, other_node in self.nodes.items():
                if node_id in other_node.connections:
                    other_node.disconnect(node_id, bidirectional=False)
                    
            # Remove from all groups
            for group_name, node_ids in list(self.node_groups.items()):
                if node_id in node_ids:
                    node_ids.remove(node_id)
                    
            # Remove from the network
            del self.nodes[node_id]
            
            return node
            
        return None
        
    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
        
    def connect_nodes(self, node1_id: str, node2_id: str, bidirectional: bool = True) -> bool:
        """Connect two nodes."""
        node1 = self.nodes.get(node1_id)
        node2 = self.nodes.get(node2_id)
        
        if node1 is None or node2 is None:
            logger.warning(f"NetworkManager cannot connect unknown nodes {node1_id}, {node2_id}")
            return False
            
        node1.connect(node2, bidirectional)
        return True
        
    def disconnect_nodes(self, node1_id: str, node2_id: str, bidirectional: bool = True) -> bool:
        """Disconnect two nodes."""
        node1 = self.nodes.get(node1_id)
        node2 = self.nodes.get(node2_id)
        
        if node1 is None or node2 is None:
            logger.warning(f"NetworkManager cannot disconnect unknown nodes {node1_id}, {node2_id}")
            return False
            
        node1.disconnect(node2_id, bidirectional)
        return True
        
    def create_group(self, group_name: str, node_ids: List[str]) -> bool:
        """Create a named group of nodes."""
        # Verify all nodes exist
        for node_id in node_ids:
            if node_id not in self.nodes:
                logger.warning(f"NetworkManager cannot create group with unknown node {node_id}")
                return False
                
        self.node_groups[group_name] = node_ids.copy()
        return True
        
    def add_to_group(self, group_name: str, node_id: str) -> bool:
        """Add a node to a group."""
        if group_name not in self.node_groups:
            logger.warning(f"NetworkManager cannot add to unknown group {group_name}")
            return False
            
        if node_id not in self.nodes:
            logger.warning(f"NetworkManager cannot add unknown node {node_id} to group")
            return False
            
        if node_id not in self.node_groups[group_name]:
            self.node_groups[group_name].append(node_id)
            
        return True
        
    def remove_from_group(self, group_name: str, node_id: str) -> bool:
        """Remove a node from a group."""
        if group_name not in self.node_groups:
            logger.warning(f"NetworkManager cannot remove from unknown group {group_name}")
            return False
            
        if node_id in self.node_groups[group_name]:
            self.node_groups[group_name].remove(node_id)
            
        return True
        
    def get_group(self, group_name: str) -> List[str]:
        """Get the node IDs in a group."""
        return self.node_groups.get(group_name, []).copy()
        
    def broadcast_to_group(self, group_name: str, message: Message) -> int:
        """Broadcast a message to all nodes in a group."""
        if group_name not in self.node_groups:
            logger.warning(f"NetworkManager cannot broadcast to unknown group {group_name}")
            return 0
            
        success_count = 0
        
        for node_id in self.node_groups[group_name]:
            node = self.nodes.get(node_id)
            if node and node.receive(message):
                success_count += 1
                
        return success_count
        
    def process_all(self) -> Dict[str, int]:
        """Process all messages in all nodes."""
        results = {}
        
        for node_id, node in self.nodes.items():
            count = node.process_all()
            results[node_id] = count
            
        return results
        
    def activate_all(self):
        """Activate all nodes."""
        for node in self.nodes.values():
            node.activate()
            
    def deactivate_all(self):
        """Deactivate all nodes."""
        for node in self.nodes.values():
            node.deactivate()
            
    def save_network(self, filepath: str):
        """Save the entire network to a file."""
        network_data = {
            "nodes": {},
            "groups": self.node_groups.copy()
        }
        
        # Save each node's state
        for node_id, node in self.nodes.items():
            node_data = {
                "node_id": node.node_id,
                "node_type": node.node_type.name,
                "connections": list(node.connections.keys()),
                "active": node.active,
                "state": node.state
            }
            
            # Add type-specific data
            if node.node_type == NodeType.PROCESSING:
                # Skip storing the process function directly
                node_data["results"] = node.results
            elif node.node_type == NodeType.STORAGE:
                node_data["storage"] = node.storage
                node_data["timestamps"] = node.timestamps
                node_data["access_counts"] = node.access_counts
                node_data["capacity"] = node.capacity
            elif node.node_type == NodeType.OUTPUT:
                # Skip storing the output handler directly
                node_data["outputs"] = node.outputs
            elif node.node_type == NodeType.FILTERING:
                # Skip storing the filter function directly
                node_data["filtered_count"] = node.filtered_count
                node_data["passed_count"] = node.passed_count
            elif node.node_type == NodeType.VISUALIZATION:
                # Skip storing the visualize function directly
                node_data["visualizations"] = node.visualizations
            elif node.node_type == NodeType.QUANTUM:
                # Skip storing the circuits and results (too complex)
                node_data["circuit_ids"] = list(node.circuits.keys())
                node_data["result_ids"] = list(node.results.keys())
            elif node.node_type == NodeType.SUPERNODE:
                node_data["subnodes"] = list(node.subnodes.keys())
                node_data["routing_table"] = {msg_type.name: targets for msg_type, targets in node.routing_table.items()}
                
            network_data["nodes"][node_id] = node_data
            
        # Save to file
        with open(filepath, 'wb') as f:
            pickle.dump(network_data, f)
            
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the network."""
        return {
            "nodes": len(self.nodes),
            "groups": {name: len(ids) for name, ids in self.node_groups.items()},
            "node_types": {node_type.name: sum(1 for node in self.nodes.values() if node.node_type == node_type) for node_type in NodeType},
            "active_nodes": sum(1 for node in self.nodes.values() if node.active),
            "connections": sum(len(node.connections) for node in self.nodes.values()) // 2,  # Divide by 2 for bidirectional connections
        }
        
    def visualize_network(self, filename: Optional[str] = None) -> nx.Graph:
        """Create a NetworkX graph of the node network and optionally save as an image."""
        G = nx.Graph()
        
        # Add nodes with attributes
        for node_id, node in self.nodes.items():
            G.add_node(node_id, 
                       type=node.node_type.name, 
                       active=node.active)
                       
        # Add edges
        for node_id, node in self.nodes.items():
            for connected_id in node.connections:
                if connected_id in self.nodes:  # Ensure the connected node exists
                    G.add_edge(node_id, connected_id)
                    
        # Save visualization if filename is provided
        if filename:
            plt.figure(figsize=(12, 8))
            
            # Create position layout
            pos = nx.spring_layout(G, seed=42)
            
            # Node colors based on type
            node_colors = []
            for node in G.nodes:
                node_type = G.nodes[node]['type']
                if node_type == NodeType.INPUT.name:
                    node_colors.append('green')
                elif node_type == NodeType.OUTPUT.name:
                    node_colors.append('red')
                elif node_type == NodeType.PROCESSING.name:
                    node_colors.append('blue')
                elif node_type == NodeType.SUPERNODE.name:
                    node_colors.append('purple')
                elif node_type == NodeType.STORAGE.name:
                    node_colors.append('orange')
                elif node_type == NodeType.FILTERING.name:
                    node_colors.append('yellow')
                elif node_type == NodeType.VISUALIZATION.name:
                    node_colors.append('cyan')
                elif node_type == NodeType.QUANTUM.name:
                    node_colors.append('magenta')
                else:
                    node_colors.append('gray')
                    
            # Node shape/size based on activity
            node_sizes = []
            for node in G.nodes:
                active = G.nodes[node]['active']
                node_sizes.append(300 if active else 100)
                
            # Draw the network
            nx.draw_networkx(G, pos, 
                             node_color=node_colors, 
                             node_size=node_sizes,
                             font_size=10, 
                             font_weight='bold')
                             
            plt.title("Brain Module Network")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(filename)
            plt.close()
            
        return G

# ============================================================================
# 3D Cube Visualization Module
# ============================================================================

class CubeVisualization:
    """3D cube visualization system for representing complex data relationships."""
    
    def __init__(self, dimensions: Tuple[int, int, int] = (10, 10, 10)):
        """Initialize the cube visualization with dimensions (x, y, z)."""
        self.dimensions = dimensions
        self.cube = np.zeros(dimensions)  # Empty cube
        self.labels = {
            'x': [f"x{i}" for i in range(dimensions[0])],
            'y': [f"y{i}" for i in range(dimensions[1])],
            'z': [f"z{i}" for i in range(dimensions[2])]
        }
        self.highlights = []  # List of (x, y, z, color) tuples
        self.connections = []  # List of ((x1,y1,z1), (x2,y2,z2), color) tuples
        self.metrics = {}  # Dictionary of metrics to display
        
    def set_value(self, x: int, y: int, z: int, value: float):
        """Set the value at a specific cube coordinate."""
        if 0 <= x < self.dimensions[0] and 0 <= y < self.dimensions[1] and 0 <= z < self.dimensions[2]:
            self.cube[x, y, z] = value
        else:
            logger.warning(f"CubeVisualization: coordinates ({x},{y},{z}) out of bounds")
            
    def get_value(self, x: int, y: int, z: int) -> float:
        """Get the value at a specific cube coordinate."""
        if 0 <= x < self.dimensions[0] and 0 <= y < self.dimensions[1] and 0 <= z < self.dimensions[2]:
            return self.cube[x, y, z]
        else:
            logger.warning(f"CubeVisualization: coordinates ({x},{y},{z}) out of bounds")
            return 0.0
            
    def fill_region(self, x_range: Tuple[int, int], y_range: Tuple[int, int], z_range: Tuple[int, int], value: float):
        """Fill a region of the cube with a value."""
        x_min, x_max = max(0, x_range[0]), min(self.dimensions[0], x_range[1])
        y_min, y_max = max(0, y_range[0]), min(self.dimensions[1], y_range[1])
        z_min, z_max = max(0, z_range[0]), min(self.dimensions[2], z_range[1])
        
        self.cube[x_min:x_max, y_min:y_max, z_min:z_max] = value
        
    def highlight_point(self, x: int, y: int, z: int, color: str = 'red'):
        """Highlight a specific point in the cube."""
        if 0 <= x < self.dimensions[0] and 0 <= y < self.dimensions[1] and 0 <= z < self.dimensions[2]:
            self.highlights.append((x, y, z, color))
        else:
            logger.warning(f"CubeVisualization: highlight coordinates ({x},{y},{z}) out of bounds")
            
    def connect_points(self, point1: Tuple[int, int, int], point2: Tuple[int, int, int], color: str = 'blue'):
        """Create a connection between two points in the cube."""
        x1, y1, z1 = point1
        x2, y2, z2 = point2
        
        if (0 <= x1 < self.dimensions[0] and 0 <= y1 < self.dimensions[1] and 0 <= z1 < self.dimensions[2] and
            0 <= x2 < self.dimensions[0] and 0 <= y2 < self.dimensions[1] and 0 <= z2 < self.dimensions[2]):
            self.connections.append((point1, point2, color))
        else:
            logger.warning(f"CubeVisualization: connection coordinates out of bounds")
            
    def set_labels(self, axis: str, labels: List[str]):
        """Set labels for a specific axis."""
        if axis in ['x', 'y', 'z'] and len(labels) == self.dimensions['xyz'.index(axis)]:
            self.labels[axis] = labels.copy()
        else:
            logger.warning(f"CubeVisualization: invalid labels for axis {axis}")
            
    def add_metric(self, name: str, value: Any):
        """Add a metric to display with the visualization."""
        self.metrics[name] = value
        
    def clear(self):
        """Clear the cube data."""
        self.cube = np.zeros(self.dimensions)
        self.highlights = []
        self.connections = []
        self.metrics = {}
        
    def visualize(self, filename: Optional[str] = None, title: str = "Cube Visualization"):
        """Generate a 3D visualization of the cube."""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Find non-zero values to visualize
        x_indices, y_indices, z_indices = np.nonzero(self.cube)
        values = self.cube[x_indices, y_indices, z_indices]
        
        # Scale sizes by values
        max_val = np.max(np.abs(values)) if len(values) > 0 else 1.0
        sizes = 100 * np.abs(values) / max_val if len(values) > 0 else []
        
        # Scale colors by values
        colors = plt.cm.viridis(values / max_val) if len(values) > 0 else []
        
        # Plot non-zero cells as points
        if len(x_indices) > 0:
            ax.scatter(x_indices, y_indices, z_indices, c=colors, s=sizes, alpha=0.7)
            
        # Plot highlighted points
        for x, y, z, color in self.highlights:
            ax.scatter([x], [y], [z], c=color, s=200, marker='*', edgecolors='black')
            
        # Plot connections
        for (x1, y1, z1), (x2, y2, z2), color in self.connections:
            ax.plot([x1, x2], [y1, y2], [z1, z2], c=color, linewidth=2, alpha=0.7)
            
        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set title with metrics
        full_title = title
        if self.metrics:
            metrics_str = ", ".join(f"{k}: {v}" for k, v in self.metrics.items())
            full_title += f"\n{metrics_str}"
        ax.set_title(full_title)
        
        # Set axis limits
        ax.set_xlim(0, self.dimensions[0] - 1)
        ax.set_ylim(0, self.dimensions[1] - 1)
        ax.set_zlim(0, self.dimensions[2] - 1)
        
        # Save to file if specified
        if filename:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()
            
    def to_networkx(self) -> nx.Graph:
        """Convert the cube to a NetworkX graph."""
        G = nx.Graph()
        
        # Add nodes for non-zero values
        x_indices, y_indices, z_indices = np.nonzero(self.cube)
        values = self.cube[x_indices, y_indices, z_indices]
        
        for i in range(len(x_indices)):
            x, y, z = x_indices[i], y_indices[i], z_indices[i]
            value = values[i]
            node_id = f"{x},{y},{z}"
            G.add_node(node_id, x=x, y=y, z=z, value=value)
            
        # Add highlighted nodes
        for x, y, z, color in self.highlights:
            node_id = f"{x},{y},{z}"
            if node_id in G.nodes:
                G.nodes[node_id]['highlighted'] = True
                G.nodes[node_id]['highlight_color'] = color
            else:
                G.add_node(node_id, x=x, y=y, z=z, value=0.0, highlighted=True, highlight_color=color)
                
        # Add connections
        for u, v, data in G.edges(data=True):
            u_data = G.nodes[u]
            v_data = G.nodes[v]
            
            if ('x' in u_data and 'y' in u_data and 'z' in u_data and
                'x' in v_data and 'y' in v_data and 'z' in v_data):
                point1 = (u_data['x'], u_data['y'], u_data['z'])
                point2 = (v_data['x'], v_data['y'], v_data['z'])
                color = data.get('color', 'blue')
                
                self.connect_points(point1, point2, color)
                
        return self
        
    def from_graph(self, G: nx.Graph):
        """Populate the cube from a NetworkX graph."""
        self.clear()
        
        # Add nodes to the cube
        for node, data in G.nodes(data=True):
            if 'x' in data and 'y' in data and 'z' in data:
                x, y, z = data['x'], data['y'], data['z']
                
                if 'value' in data:
                    self.set_value(x, y, z, data['value'])
                    
                if data.get('highlighted', False):
                    color = data.get('highlight_color', 'red')
                    self.highlight_point(x, y, z, color)
                    
        # Add edges as connections
        for u, v, data in G.edges(data=True):
            u_data = G.nodes[u]
            v_data = G.nodes[v]
            
            if ('x' in u_data and 'y' in u_data and 'z' in u_data and
                'x' in v_data and 'y' in v_data and 'z' in v_data):
                point1 = (u_data['x'], u_data['y'], u_data['z'])
                point2 = (v_data['x'], v_data['y'], v_data['z'])
                color = data.get('color', 'blue')
                
                self.connect_points(point1, point2, color)
                
        return self

class CubeCluster:
    """Clustering analysis for cube data."""
    
    def __init__(self, cube: CubeVisualization):
        """Initialize with a cube visualization."""
        self.cube = cube
        
    def identify_clusters(self, threshold: float = 0.5, min_size: int = 2) -> List[List[Tuple[int, int, int]]]:
        """Identify clusters of related points in the cube."""
        # Create a graph of connected points
        G = nx.Graph()
        
        # Add nodes for points above threshold
        x_indices, y_indices, z_indices = np.nonzero(self.cube.cube >= threshold)
        
        for i in range(len(x_indices)):
            x, y, z = x_indices[i], y_indices[i], z_indices[i]
            G.add_node((x, y, z), value=self.cube.cube[x, y, z])
            
        # Connect adjacent nodes
        for i in range(len(x_indices)):
            x1, y1, z1 = x_indices[i], y_indices[i], z_indices[i]
            for j in range(i + 1, len(x_indices)):
                x2, y2, z2 = x_indices[j], y_indices[j], z_indices[j]
                
                # Check if points are adjacent (Manhattan distance = 1)
                if abs(x1 - x2) + abs(y1 - y2) + abs(z1 - z2) == 1:
                    G.add_edge((x1, y1, z1), (x2, y2, z2))
                    
        # Find connected components (clusters)
        clusters = list(nx.connected_components(G))
        
        # Filter by minimum size
        clusters = [list(cluster) for cluster in clusters if len(cluster) >= min_size]
        
        return clusters
        
    def highlight_clusters(self, threshold: float = 0.5, min_size: int = 2, colors: List[str] = None):
        """Identify and highlight clusters in the cube."""
        clusters = self.identify_clusters(threshold, min_size)
        
        if not colors:
            colors = ['red', 'green', 'blue', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
            
        # Highlight clusters with different colors
        for i, cluster in enumerate(clusters):
            color = colors[i % len(colors)]
            
            # Highlight all points in the cluster
            for x, y, z in cluster:
                self.cube.highlight_point(x, y, z, color)
                
            # Connect all points in the cluster
            for j, point1 in enumerate(cluster):
                for point2 in cluster[j+1:]:
                    self.cube.connect_points(point1, point2, color)
                    
        return clusters
        
    def calculate_cluster_metrics(self, clusters: List[List[Tuple[int, int, int]]]) -> Dict[str, Any]:
        """Calculate metrics for the identified clusters."""
        metrics = {
            "num_clusters": len(clusters),
            "cluster_sizes": [len(cluster) for cluster in clusters],
            "avg_cluster_size": sum(len(cluster) for cluster in clusters) / len(clusters) if clusters else 0,
            "max_cluster_size": max(len(cluster) for cluster in clusters) if clusters else 0,
            "total_points": sum(len(cluster) for cluster in clusters),
            "cluster_densities": [],
            "cluster_centers": []
        }
        
        for cluster in clusters:
            # Calculate center
            center_x = sum(p[0] for p in cluster) / len(cluster)
            center_y = sum(p[1] for p in cluster) / len(cluster)
            center_z = sum(p[2] for p in cluster) / len(cluster)
            metrics["cluster_centers"].append((center_x, center_y, center_z))
            
            # Calculate density (percentage of points in the bounding box)
            min_x = min(p[0] for p in cluster)
            max_x = max(p[0] for p in cluster)
            min_y = min(p[1] for p in cluster)
            max_y = max(p[1] for p in cluster)
            min_z = min(p[2] for p in cluster)
            max_z = max(p[2] for p in cluster)
            
            box_volume = (max_x - min_x + 1) * (max_y - min_y + 1) * (max_z - min_z + 1)
            density = len(cluster) / box_volume if box_volume > 0 else 1.0
            metrics["cluster_densities"].append(density)
            
        return metrics
        
    def find_paths_between_clusters(self, clusters: List[List[Tuple[int, int, int]]], 
                                  max_distance: int = 5) -> List[Tuple[int, int, List[Tuple[int, int, int]]]]:
        """Find shortest paths between clusters."""
        paths = []
        
        # Create a graph with all points
        G = nx.Graph()
        
        # Add all points as nodes
        x_indices, y_indices, z_indices = np.nonzero(self.cube.cube)
        for i in range(len(x_indices)):
            x, y, z = x_indices[i], y_indices[i], z_indices[i]
            G.add_node((x, y, z), value=self.cube.cube[x, y, z])
            
        # Connect adjacent nodes
        for i in range(len(x_indices)):
            x1, y1, z1 = x_indices[i], y_indices[i], z_indices[i]
            for j in range(i + 1, len(x_indices)):
                x2, y2, z2 = x_indices[j], y_indices[j], z_indices[j]
                
                # Check if points are close (Manhattan distance <= max_distance)
                distance = abs(x1 - x2) + abs(y1 - y2) + abs(z1 - z2)
                if distance <= max_distance:
                    # Weight by inverse of the values
                    weight = 2.0 - (self.cube.cube[x1, y1, z1] + self.cube.cube[x2, y2, z2]) / 2.0
                    G.add_edge((x1, y1, z1), (x2, y2, z2), weight=weight)
                    
        # Find paths between clusters
        for i, cluster1 in enumerate(clusters):
            for j, cluster2 in enumerate(clusters):
                if i < j:  # Only process each pair once
                    # Find the shortest path between any two points in the clusters
                    shortest_path = None
                    min_length = float('inf')
                    
                    for point1 in cluster1:
                        for point2 in cluster2:
                            try:
                                path = nx.shortest_path(G, point1, point2, weight='weight')
                                if len(path) < min_length:
                                    min_length = len(path)
                                    shortest_path = path
                            except nx.NetworkXNoPath:
                                pass
                                
                    if shortest_path:
                        paths.append((i, j, shortest_path))
                        
        return paths
        
    def visualize_cluster_connections(self, paths: List[Tuple[int, int, List[Tuple[int, int, int]]]],
                                     colors: List[str] = None):
        """Visualize paths between clusters."""
        if not colors:
            colors = ['limegreen', 'cyan', 'yellow', 'white', 'pink', 'orange', 'purple', 'brown']
            
        # Highlight paths with different colors
        for i, (cluster1_idx, cluster2_idx, path) in enumerate(paths):
            color = colors[i % len(colors)]
            
            # Connect all points in the path
            for j in range(len(path) - 1):
                self.cube.connect_points(path[j], path[j+1], color)
                
        return self

# ============================================================================
# Graph-Based Pattern Recognition Module
# ============================================================================

class PatternGraph:
    """Graph-based pattern recognition module."""
    
    def __init__(self):
        """Initialize the pattern graph."""
        self.graph = nx.DiGraph()
        self.patterns = {}
        self.pattern_instances = {}
        self.match_thresholds = {}
        
    def add_node(self, node_id: str, attributes: Dict[str, Any] = None):
        """Add a node to the pattern graph."""
        self.graph.add_node(node_id, **(attributes or {}))
        
    def add_edge(self, source_id: str, target_id: str, attributes: Dict[str, Any] = None):
        """Add an edge to the pattern graph."""
        self.graph.add_edge(source_id, target_id, **(attributes or {}))
        
    def define_pattern(self, pattern_id: str, node_ids: List[str], edge_constraints: List[Tuple[int, int, Dict[str, Any]]],
                      match_threshold: float = 0.8):
        """
        Define a pattern in the graph.
        
        Args:
            pattern_id: Unique identifier for the pattern
            node_ids: List of node IDs that form the pattern
            edge_constraints: List of (source_idx, target_idx, attributes) tuples defining edge constraints
            match_threshold: Threshold for considering a match (0.0 to 1.0)
        """
        # Create the pattern subgraph
        pattern = nx.DiGraph()
        
        # Add nodes
        for i, node_id in enumerate(node_ids):
        match_threshold = self.match_thresholds[pattern_id]
        
        # Use VF2 algorithm for isomorphism matching
        matcher = nx.algorithms.isomorphism.DiGraphMatcher(
            self.graph, pattern_graph,
            node_match=lambda n1, n2: self._node_match(n1, n2, match_threshold),
            edge_match=lambda e1, e2: self._edge_match(e1, e2, match_threshold)
        )
        
        # Find all isomorphic subgraphs
        instances = []
        for mapping in matcher.subgraph_isomorphisms_iter():
            # Convert the mapping to a list of node IDs
            instance = [mapping[i] for i in range(len(pattern["nodes"]))]
            instances.append(instance)
            
        # Store the instances
        self.pattern_instances[pattern_id] = instances
        
        return instances
        
    def _node_match(self, n1: Dict[str, Any], n2: Dict[str, Any], threshold: float) -> bool:
        """Check if two nodes match based on their attributes."""
        # If either node has no attributes, consider it a wildcard
        if not n1 or not n2:
            return True
            
        # Count matching attributes
        matches = 0
        total = 0
        
        for attr, value in n2.items():
            if attr in n1:
                total += 1
                if n1[attr] == value:
                    matches += 1
                    
        # If no attributes to compare, consider it a match
        if total == 0:
            return True
            
        # Check if the match ratio exceeds the threshold
        return matches / total >= threshold
        
    def _edge_match(self, e1: Dict[str, Any], e2: Dict[str, Any], threshold: float) -> bool:
        """Check if two edges match based on their attributes."""
        # If either edge has no attributes, consider it a wildcard
        if not e1 or not e2:
            return True
            
        # Count matching attributes
        matches = 0
        total = 0
        
        for attr, value in e2.items():
            if attr in e1:
                total += 1
                if e1[attr] == value:
                    matches += 1
                    
        # If no attributes to compare, consider it a match
        if total == 0:
            return True
            
        # Check if the match ratio exceeds the threshold
        return matches / total >= threshold
        
    def get_pattern_statistics(self, pattern_id: str) -> Dict[str, Any]:
        """Get statistics about a pattern and its instances."""
        if pattern_id not in self.patterns or pattern_id not in self.pattern_instances:
            logger.warning(f"PatternGraph: unknown pattern {pattern_id}")
            return {}
            
        instances = self.pattern_instances[pattern_id]
        
        stats = {
            "pattern_id": pattern_id,
            "num_instances": len(instances),
            "instance_nodes": [instance for instance in instances],
            "avg_distance": 0.0,
            "centrality": {},
            "overlap": {}
        }
        
        if instances:
            # Calculate average distance between instances
            distances = []
            for i, instance1 in enumerate(instances):
                for j, instance2 in enumerate(instances):
                    if i < j:
                        # Calculate Jaccard distance between instances
                        set1 = set(instance1)
                        set2 = set(instance2)
                        jaccard = len(set1.intersection(set2)) / len(set1.union(set2))
                        distances.append(1.0 - jaccard)  # Convert similarity to distance
                        
            stats["avg_distance"] = sum(distances) / len(distances) if distances else 0.0
            
            # Calculate node centrality within pattern instances
            all_import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.fftpack import fft, ifft
from typing import Dict, List, Tuple, Set, Optional, Union, Callable, Any
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings
import logging
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import os
import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import heapq

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# Quantum Simulator Module
# ============================================================================

class QuantumState:
    def __init__(self, n_qubits: int):
        """Initialize a quantum state with n qubits."""
        self.n_qubits = n_qubits
        self.state_dim = 2 ** n_qubits
        self.amplitudes = np.zeros(self.state_dim, dtype=np.complex128)
        self.amplitudes[0] = 1.0  # Initialize to |0...0⟩
        
    def apply_gate(self, gate_matrix: np.ndarray, target_qubits: List[int]):
        """Apply a quantum gate to specified qubits."""
        # Calculate the full transformation matrix
        full_matrix = self._expand_gate(gate_matrix, target_qubits)
        # Apply the gate
        self.amplitudes = full_matrix @ self.amplitudes
        # Normalize to handle floating-point errors
        self.normalize()
        
    def _expand_gate(self, gate_matrix: np.ndarray, target_qubits: List[int]) -> np.ndarray:
        """Expand the gate matrix to act on the full Hilbert space."""
        # Implementation of tensor product expansion for quantum gates
        sorted_targets = sorted(target_qubits)
        n_targets = len(sorted_targets)
        
        # Check if the gate matrix matches the number of target qubits
        if gate_matrix.shape != (2**n_targets, 2**n_targets):
            raise ValueError(f"Gate matrix shape {gate_matrix.shape} doesn't match for {n_targets} qubits")
            
        # Build the full matrix using sparse matrices for efficiency
        indices = list(range(self.n_qubits))
        for i, target in enumerate(sorted_targets):
            indices.remove(target)
        
        # Permute qubits to bring targets to the beginning
        permutation = sorted_targets + indices
        inv_permutation = [0] * self.n_qubits
        for i, p in enumerate(permutation):
            inv_permutation[p] = i
            
        # Calculate permutation matrices
        perm = np.zeros((self.state_dim, self.state_dim), dtype=np.complex128)
        inv_perm = np.zeros((self.state_dim, self.state_dim), dtype=np.complex128)
        
        for i in range(self.state_dim):
            # Convert i to binary, permute bits, convert back to decimal
            bin_i = format(i, f'0{self.n_qubits}b')
            perm_bits = ''.join(bin_i[inv_permutation[j]] for j in range(self.n_qubits))
            perm_i = int(perm_bits, 2)
            perm[perm_i, i] = 1
            inv_perm[i, perm_i] = 1
            
        # Create the expanded gate
        expanded_gate = np.identity(2**(self.n_qubits - n_targets), dtype=np.complex128)
        expanded_gate = np.kron(gate_matrix, expanded_gate)
        
        # Apply the permutations
        return inv_perm @ expanded_gate @ perm
    
    def normalize(self):
        """Normalize the quantum state."""
        norm = np.sqrt(np.sum(np.abs(self.amplitudes) ** 2))
        if norm > 1e-10:  # Avoid division by near-zero
            self.amplitudes /= norm
            
    def measure(self, collapse: bool = True) -> Tuple[int, float]:
        """Measure the quantum state, optionally collapsing it."""
        probabilities = np.abs(self.amplitudes) ** 2
        result = np.random.choice(self.state_dim, p=probabilities)
        
        if collapse:
            # Collapse the state to the measured basis state
            self.amplitudes = np.zeros_like(self.amplitudes)
            self.amplitudes[result] = 1.0
            
        # Return the measurement result and its probability
        return result, probabilities[result]
    
    def measure_qubit(self, qubit_index: int, collapse: bool = True) -> Tuple[int, float]:
        """Measure a specific qubit."""
        if qubit_index >= self.n_qubits:
            raise ValueError(f"Qubit index {qubit_index} out of range for {self.n_qubits} qubits")
            
        # Calculate probabilities for qubit being 0 or 1
        prob_0 = 0.0
        prob_1 = 0.0
        
        for i in range(self.state_dim):
            # Check if the qubit_index bit is 0 or 1
            if (i >> qubit_index) & 1 == 0:
                prob_0 += np.abs(self.amplitudes[i]) ** 2
            else:
                prob_1 += np.abs(self.amplitudes[i]) ** 2
                
        # Normalize probabilities (in case of floating-point errors)
        total_prob = prob_0 + prob_1
        prob_0 /= total_prob
        prob_1 /= total_prob
        
        # Determine the result
        if np.random.random() < prob_0:
            result = 0
            prob = prob_0
        else:
            result = 1
            prob = prob_1
            
        if collapse:
            # Collapse the state
            new_amplitudes = np.zeros_like(self.amplitudes)
            normalization = 0.0
            
            for i in range(self.state_dim):
                bit_val = (i >> qubit_index) & 1
                if bit_val == result:
                    new_amplitudes[i] = self.amplitudes[i]
                    normalization += np.abs(self.amplitudes[i]) ** 2
                    
            new_amplitudes /= np.sqrt(normalization)
            self.amplitudes = new_amplitudes
            
        return result, prob
        
    def entangle(self, other_state: 'QuantumState') -> 'QuantumState':
        """Entangle this quantum state with another one."""
        total_qubits = self.n_qubits + other_state.n_qubits
        entangled = QuantumState(total_qubits)
        
        # Tensor product of states
        entangled.amplitudes = np.kron(self.amplitudes, other_state.amplitudes)
        return entangled

    def density_matrix(self) -> np.ndarray:
        """Calculate the density matrix representation of the state."""
        return np.outer(self.amplitudes, np.conjugate(self.amplitudes))
    
    def partial_trace(self, keep_qubits: List[int]) -> np.ndarray:
        """Perform a partial trace, keeping only specified qubits."""
        trace_qubits = [i for i in range(self.n_qubits) if i not in keep_qubits]
        
        # Calculate dimensions
        keep_dim = 2 ** len(keep_qubits)
        trace_dim = 2 ** len(trace_qubits)
        
        # Initialize reduced density matrix
        reduced_dm = np.zeros((keep_dim, keep_dim), dtype=np.complex128)
        
        # Convert to density matrix
        full_dm = self.density_matrix()
        
        # Perform partial trace
        for i in range(trace_dim):
            bin_i = format(i, f'0{len(trace_qubits)}b')
            
            for j in range(keep_dim):
                for k in range(keep_dim):
                    # Calculate full indices
                    idx_j = self._combine_indices(j, i, keep_qubits, trace_qubits)
                    idx_k = self._combine_indices(k, i, keep_qubits, trace_qubits)
                    
                    # Add to reduced density matrix
                    reduced_dm[j, k] += full_dm[idx_j, idx_k]
                    
        return reduced_dm
    
    def _combine_indices(self, keep_idx: int, trace_idx: int, 
                         keep_qubits: List[int], trace_qubits: List[int]) -> int:
        """Combine separated indices into a full Hilbert space index."""
        keep_bits = format(keep_idx, f'0{len(keep_qubits)}b')
        trace_bits = format(trace_idx, f'0{len(trace_qubits)}b')
        
        # Combine bits
        full_bits = ['0'] * self.n_qubits
        for i, qubit in enumerate(keep_qubits):
            full_bits[qubit] = keep_bits[i]
        for i, qubit in enumerate(trace_qubits):
            full_bits[qubit] = trace_bits[i]
            
        # Convert to decimal
        return int(''.join(full_bits), 2)

class QuantumGates:
    """Common quantum gates."""
    
    @staticmethod
    def I() -> np.ndarray:
        """Identity gate."""
        return np.array([[1, 0], [0, 1]], dtype=np.complex128)
        
    @staticmethod
    def X() -> np.ndarray:
        """Pauli-X (NOT) gate."""
        return np.array([[0, 1], [1, 0]], dtype=np.complex128)
        
    @staticmethod
    def Y() -> np.ndarray:
        """Pauli-Y gate."""
        return np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        
    @staticmethod
    def Z() -> np.ndarray:
        """Pauli-Z gate."""
        return np.array([[1, 0], [0, -1]], dtype=np.complex128)
        
    @staticmethod
    def H() -> np.ndarray:
        """Hadamard gate."""
        return np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
        
    @staticmethod
    def S() -> np.ndarray:
        """Phase gate."""
        return np.array([[1, 0], [0, 1j]], dtype=np.complex128)
        
    @staticmethod
    def T() -> np.ndarray:
        """T gate (π/8 gate)."""
        return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)
        
    @staticmethod
    def CNOT() -> np.ndarray:
        """Controlled-NOT gate."""
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=np.complex128)
        
    @staticmethod
    def SWAP() -> np.ndarray:
        """SWAP gate."""
        return np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.complex128)
    
    @staticmethod
    def Rx(theta: float) -> np.ndarray:
        """Rotation around X-axis."""
        return np.array([
            [np.cos(theta/2), -1j * np.sin(theta/2)],
            [-1j * np.sin(theta/2), np.cos(theta/2)]
        ], dtype=np.complex128)
        
    @staticmethod
    def Ry(theta: float) -> np.ndarray:
        """Rotation around Y-axis."""
        return np.array([
            [np.cos(theta/2), -np.sin(theta/2)],
            [np.sin(theta/2), np.cos(theta/2)]
        ], dtype=np.complex128)
        
    @staticmethod
    def Rz(theta: float) -> np.ndarray:
        """Rotation around Z-axis."""
        return np.array([
            [np.exp(-1j * theta/2), 0],
            [0, np.exp(1j * theta/2)]
        ], dtype=np.complex128)
        
    @staticmethod
    def CZ() -> np.ndarray:
        """Controlled-Z gate."""
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1]
        ], dtype=np.complex128)
        
    @staticmethod
    def Toffoli() -> np.ndarray:
        """Toffoli (CCNOT) gate."""
        toffoli = np.eye(8, dtype=np.complex128)
        # Swap the last two elements of the last two rows
        toffoli[6, 6] = 0
        toffoli[6, 7] = 1
        toffoli[7, 6] = 1
        toffoli[7, 7] = 0
        return toffoli

class QuantumRegister:
    """A quantum register composed of qubits."""
    
    def __init__(self, n_qubits: int):
        """Initialize a quantum register with n qubits."""
        self.state = QuantumState(n_qubits)
        self.n_qubits = n_qubits
        
    def apply_gate(self, gate: Union[str, np.ndarray], 
                 target_qubits: List[int], 
                 params: Optional[List[float]] = None):
        """Apply a gate to target qubits."""
        # Get the gate matrix
        if isinstance(gate, str):
            gate_matrix = self._get_gate_matrix(gate, params)
        else:
            gate_matrix = gate
            
        # Apply the gate
        self.state.apply_gate(gate_matrix, target_qubits)
        
    def _get_gate_matrix(self, gate_name: str, params: Optional[List[float]] = None) -> np.ndarray:
        """Get the gate matrix by name."""
        gates = QuantumGates()
        
        if gate_name == 'I':
            return gates.I()
        elif gate_name == 'X':
            return gates.X()
        elif gate_name == 'Y':
            return gates.Y()
        elif gate_name == 'Z':
            return gates.Z()
        elif gate_name == 'H':
            return gates.H()
        elif gate_name == 'S':
            return gates.S()
        elif gate_name == 'T':
            return gates.T()
        elif gate_name == 'CNOT':
            return gates.CNOT()
        elif gate_name == 'SWAP':
            return gates.SWAP()
        elif gate_name == 'CZ':
            return gates.CZ()
        elif gate_name == 'Toffoli':
            return gates.Toffoli()
        elif gate_name == 'Rx':
            if params is None or len(params) < 1:
                raise ValueError("Rx gate requires a theta parameter")
            return gates.Rx(params[0])
        elif gate_name == 'Ry':
            if params is None or len(params) < 1:
                raise ValueError("Ry gate requires a theta parameter")
            return gates.Ry(params[0])
        elif gate_name == 'Rz':
            if params is None or len(params) < 1:
                raise ValueError("Rz gate requires a theta parameter")
            return gates.Rz(params[0])
        else:
            raise ValueError(f"Unknown gate: {gate_name}")
            
    def measure(self, collapse: bool = True) -> Tuple[int, float]:
        """Measure the entire register."""
        return self.state.measure(collapse)
        
    def measure_qubit(self, qubit_index: int, collapse: bool = True) -> Tuple[int, float]:
        """Measure a specific qubit."""
        return self.state.measure_qubit(qubit_index, collapse)
        
    def get_probabilities(self) -> np.ndarray:
        """Get measurement probabilities for all states."""
        return np.abs(self.state.amplitudes) ** 2
        
    def get_statevector(self) -> np.ndarray:
        """Get the state vector."""
        return self.state.amplitudes.copy()
        
    def set_statevector(self, statevector: np.ndarray):
        """Set the state vector directly."""
        if len(statevector) != self.state.state_dim:
            raise ValueError(f"State vector dimension {len(statevector)} doesn't match register dimension {self.state.state_dim}")
        self.state.amplitudes = statevector.copy()
        self.state.normalize()
        
    def reset(self):
        """Reset the register to |0...0⟩ state."""
        self.state.amplitudes = np.zeros_like(self.state.amplitudes)
        self.state.amplitudes[0] = 1.0

class QuantumCircuit:
    """A quantum circuit for gate-based quantum computation."""
    
    def __init__(self, n_qubits: int):
        """Initialize a quantum circuit with n qubits."""
        self.n_qubits = n_qubits
        self.register = QuantumRegister(n_qubits)
        self.operations = []  # List of operations to execute
        
    def add_gate(self, gate: str, target_qubits: List[int], params: Optional[List[float]] = None):
        """Add a gate to the circuit."""
        self.operations.append(('gate', gate, target_qubits, params))
        return self
        
    def add_measurement(self, target_qubits: List[int]):
        """Add a measurement operation."""
        self.operations.append(('measure', target_qubits))
        return self
        
    def reset(self):
        """Reset the circuit's register and clear operations."""
        self.register.reset()
        self.operations = []
        return self
        
    def x(self, qubit: int):
        """Apply X gate to a qubit."""
        return self.add_gate('X', [qubit])
        
    def y(self, qubit: int):
        """Apply Y gate to a qubit."""
        return self.add_gate('Y', [qubit])
        
    def z(self, qubit: int):
        """Apply Z gate to a qubit."""
        return self.add_gate('Z', [qubit])
        
    def h(self, qubit: int):
        """Apply Hadamard gate to a qubit."""
        return self.add_gate('H', [qubit])
        
    def s(self, qubit: int):
        """Apply S gate to a qubit."""
        return self.add_gate('S', [qubit])
        
    def t(self, qubit: int):
        """Apply T gate to a qubit."""
        return self.add_gate('T', [qubit])
        
    def rx(self, qubit: int, theta: float):
        """Apply Rx gate to a qubit."""
        return self.add_gate('Rx', [qubit], [theta])
        
    def ry(self, qubit: int, theta: float):
        """Apply Ry gate to a qubit."""
        return self.add_gate('Ry', [qubit], [theta])
        
    def rz(self, qubit: int, theta: float):
        """Apply Rz gate to a qubit."""
        return self.add_gate('Rz', [qubit], [theta])
        
    def cnot(self, control: int, target: int):
        """Apply CNOT gate between control and target qubits."""
        return self.add_gate('CNOT', [control, target])
        
    def swap(self, qubit1: int, qubit2: int):
        """Apply SWAP gate between two qubits."""
        return self.add_gate('SWAP', [qubit1, qubit2])
        
    def cz(self, control: int, target: int):
        """Apply CZ gate between control and target qubits."""
        return self.add_gate('CZ', [control, target])
        
    def toffoli(self, control1: int, control2: int, target: int):
        """Apply Toffoli (CCNOT) gate."""
        return self.add_gate('Toffoli', [control1, control2, target])
        
    def barrier(self):
        """Add a barrier (no operation, just for visualization)."""
        self.operations.append(('barrier',))
        return self
        
    def run(self, shots: int = 1) -> Dict[str, int]:
        """Run the circuit for a specified number of shots."""
        # Reset the register
        self.register.reset()
        
        # Dictionary to store measurement results
        results = {}
        
        for _ in range(shots):
            # Reset for each shot
            self.register.reset()
            
            # Execute all operations
            measurement_results = {}
            
            for op in self.operations:
                if op[0] == 'gate':
                    _, gate, target_qubits, params = op
                    self.register.apply_gate(gate, target_qubits, params)
                elif op[0] == 'measure':
                    _, target_qubits = op
                    for qubit in target_qubits:
                        result, _ = self.register.measure_qubit(qubit)
                        measurement_results[qubit] = result
                        
            # Format the result as a binary string
            if measurement_results:
                sorted_qubits = sorted(measurement_results.keys())
                result_str = ''.join(str(measurement_results[q]) for q in sorted_qubits)
                
                if result_str in results:
                    results[result_str] += 1
                else:
                    results[result_str] = 1
                    
        # Convert counts to probabilities
        probabilities = {k: v / shots for k, v in results.items()}
        return probabilities
        
    def get_statevector(self) -> np.ndarray:
        """Get the final state vector after circuit execution (without measurements)."""
        # Reset the register
        self.register.reset()
        
        # Execute all gate operations (skip measurements)
        for op in self.operations:
            if op[0] == 'gate':
                _, gate, target_qubits, params = op
                self.register.apply_gate(gate, target_qubits, params)
                    
        return self.register.get_statevector()
        
    def depth(self) -> int:
        """Calculate the circuit depth."""
        depth = 0
        qubit_layers = [-1] * self.n_qubits
        
        for op in self.operations:
            if op[0] == 'gate':
                _, _, target_qubits, _ = op
                
                # Find the most recent layer among target qubits
                latest_layer = max(qubit_layers[q] for q in target_qubits)
                
                # Assign this operation to the next layer
                new_layer = latest_layer + 1
                
                # Update depth if needed
                depth = max(depth, new_layer + 1)
                
                # Update the layer for all target qubits
                for q in target_qubits:
                    qubit_layers[q] = new_layer
            elif op[0] == 'barrier':
                # Synchronize all qubits
                max_layer = max(qubit_layers)
                qubit_layers = [max_layer] * self.n_qubits
                
        return depth
        
    def to_matrix(self) -> np.ndarray:
        """Convert the circuit to a unitary matrix (without measurements)."""
        # Start with identity matrix
        dim = 2 ** self.n_qubits
        matrix = np.eye(dim, dtype=np.complex128)
        
        # Apply each gate operation in reverse order (matrix multiplication order)
        for op in reversed(self.operations):
            if op[0] == 'gate':
                _, gate, target_qubits, params = op
                
                # Get the gate matrix
                gate_matrix = self.register._get_gate_matrix(gate, params)
                
                # Expand the gate to act on the full Hilbert space
                expanded_gate = QuantumState(self.n_qubits)._expand_gate(gate_matrix, target_qubits)
                
                # Apply the gate
                matrix = expanded_gate @ matrix
                
        return matrix

class QuantumSimulator:
    """Quantum simulator for quantum-inspired computation."""
    
    def __init__(self, max_qubits: int = 20):
        """Initialize the quantum simulator with a maximum number of qubits."""
        self.max_qubits = max_qubits
        
    def create_circuit(self, n_qubits: int) -> QuantumCircuit:
        """Create a new quantum circuit."""
        if n_qubits > self.max_qubits:
            raise ValueError(f"Number of qubits {n_qubits} exceeds maximum {self.max_qubits}")
        return QuantumCircuit(n_qubits)
        
    def run_circuit(self, circuit: QuantumCircuit, shots: int = 1024) -> Dict[str, int]:
        """Run a quantum circuit for the specified number of shots."""
        return circuit.run(shots)
        
    def get_statevector(self, circuit: QuantumCircuit) -> np.ndarray:
        """Get the statevector from a circuit."""
        return circuit.get_statevector()
        
    def get_unitary(self, circuit: QuantumCircuit) -> np.ndarray:
        """Get the unitary matrix representing the circuit."""
        return circuit.to_matrix()
        
    def quantum_fourier_transform(self, n_qubits: int) -> QuantumCircuit:
        """Create a Quantum Fourier Transform circuit."""
        qft = QuantumCircuit(n_qubits)
        
        for i in range(n_qubits):
            # Apply Hadamard
            qft.h(i)
            
            # Apply controlled rotations
            for j in range(i + 1, n_qubits):
                theta = 2 * np.pi / (2 ** (j - i))
                qft.add_gate('Rz', [j], [theta])
                
        # Swap qubits (bit reversal)
        for i in range(n_qubits // 2):
            qft.swap(i, n_qubits - i - 1)
            
        return qft
        
    def simulate_grover(self, n_qubits: int, oracle_function: Callable[[int], bool], 
                       iterations: Optional[int] = None) -> QuantumCircuit:
        """
        Simulate Grover's algorithm with a given oracle function.
        
        Args:
            n_qubits: Number of qubits in the search space
            oracle_function: Function that returns True for target items
            iterations: Number of Grover iterations (defaults to optimal)
            
        Returns:
            The circuit after execution
        """
        # Create the circuit
        circuit = QuantumCircuit(n_qubits)
        
        # Step 1: Initialize in superposition
        for i in range(n_qubits):
            circuit.h(i)
            
        # Calculate optimal number of iterations if not specified
        N = 2 ** n_qubits
        if iterations is None:
            iterations = int(np.floor(np.pi/4 * np.sqrt(N)))
            
        # Implement oracle as a diagonal matrix
        oracle_matrix = np.eye(N, dtype=np.complex128)
        for i in range(N):
            if oracle_function(i):
                oracle_matrix[i, i] = -1  # Phase flip for marked items
                
        # Diffusion operator as a matrix
        diffusion = np.full((N, N), 2/N, dtype=np.complex128) - np.eye(N, dtype=np.complex128)
        
        # Step 2: Apply Grover iterations
        for _ in range(iterations):
            # Oracle
            circuit.add_gate(oracle_matrix, list(range(n_qubits)))
            
            # Diffusion
            circuit.add_gate(diffusion, list(range(n_qubits)))
            
        return circuit
        
    def simulate_quantum_phase_estimation(self, 
                                        n_counting_qubits: int,
                                        unitary_matrix: np.ndarray) -> QuantumCircuit:
        """
        Simulate Quantum Phase Estimation algorithm.
        
        Args:
            n_counting_qubits: Number of qubits for the counting register
            unitary_matrix: Unitary matrix whose eigenvalue we want to estimate
            
        Returns:
            The circuit after execution
        """
        # Determine the size of the unitary matrix
        unitary_size = unitary_matrix.shape[0]
        n_target_qubits = int(np.log2(unitary_size))
        
        if 2**n_target_qubits != unitary_size:
            raise ValueError(f"Unitary matrix size {unitary_size} is not a power of 2")
            
        total_qubits = n_counting_qubits + n_target_qubits
        circuit = QuantumCircuit(total_qubits)
        
        # Initialize counting qubits in superposition
        for i in range(n_counting_qubits):
            circuit.h(i)
            
        # Apply controlled-U operations
        for i in range(n_counting_qubits):
            # Create controlled version of U^(2^i)
            power = 2 ** i
            repeated_unitary = np.linalg.matrix_power(unitary_matrix, power)
            
            # Convert to controlled operation
            controlled_u = np.eye(2 * unitary_size, dtype=np.complex128)
            controlled_u[unitary_size:, unitary_size:] = repeated_unitary
            
            # Apply controlled operation
            control_qubit = n_counting_qubits - 1 - i  # Reversed order
            target_qubits = list(range(n_counting_qubits, total_qubits))
            circuit.add_gate(controlled_u, [control_qubit] + target_qubits)
            
        # Apply inverse QFT to counting qubits
        inverse_qft = self.quantum_fourier_transform(n_counting_qubits)
        inverse_qft_matrix = inverse_qft.to_matrix().conj().T  # Hermitian conjugate for inverse
        
        counting_qubits = list(range(n_counting_qubits))
        circuit.add_gate(inverse_qft_matrix, counting_qubits)
        
        return circuit
    
    def simulate_shor(self, N: int, a: int) -> Dict[str, Any]:
        """
        Simulate Shor's factoring algorithm for factoring N using coprime a.
        
        Args:
            N: The number to factor
            a: A coprime to N (gcd(a, N) = 1)
            
        Returns:
            Dictionary with results of the simulation
        """
        import math
        
        # Check if a and N are coprime
        if math.gcd(a, N) != 1:
            raise ValueError(f"a={a} and N={N} are not coprime")
            
        # Determine the number of qubits needed
        n_count = 2 * math.ceil(math.log2(N))  # Counting register
        n_work = math.ceil(math.log2(N))       # Work register
        
        # Create circuit and registers
        circuit = QuantumCircuit(n_count + n_work)
        
        # Initialize counting register in superposition
        for i in range(n_count):
            circuit.h(i)
            
        # Initialize work register to |1⟩
        # (skip, as |0...01⟩ requires no operations starting from |0...0⟩)
        work_start = n_count
        circuit.x(work_start)  # Set the least significant bit to 1
        
        # Apply modular exponentiation: |x⟩|1⟩ → |x⟩|a^x mod N⟩
        # This is a complex operation typically simulated classically
        # For each x from 0 to 2^n_count - 1:
        x_dim = 2**n_count
        N_dim = 2**n_work
        
        modexp_matrix = np.zeros((x_dim * N_dim, x_dim * N_dim), dtype=np.complex128)
        
        for x in range(x_dim):
            ax_mod_N = pow(a, x, N)
            
            # For each input state |x⟩|y⟩
            for y in range(N_dim):
                # Map to |x⟩|(y * a^x) mod N⟩
                y_new = (y * ax_mod_N) % N_dim
                
                # Calculate indices in the full state space
                idx_in = x * N_dim + y
                idx_out = x * N_dim + y_new
                
                modexp_matrix[idx_out, idx_in] = 1
                
        # Apply the modular exponentiation unitary
        all_qubits = list(range(n_count + n_work))
        circuit.add_gate(modexp_matrix, all_qubits)
        
        # Apply inverse QFT to the counting register
        inverse_qft = self.quantum_fourier_transform(n_count)
        inverse_qft_matrix = inverse_qft.to_matrix().conj().T
        
        counting_qubits = list(range(n_count))
        circuit.add_gate(inverse_qft_matrix, counting_qubits)
        
        # Measure the counting register
        circuit.add_measurement(counting_qubits)
        
        # Run the circuit
        results = circuit.run(shots=1000)
        
        # Post-process results to find the period
        periods = []
        
        for result_str, count in results.items():
            # Convert binary string to integer
            measured_value = int(result_str, 2)
            
            # Convert to a fraction using continued fractions
            fraction = self._continued_fraction_expansion(measured_value, 2**n_count)
            
            if fraction[1] < N and fraction[1] > 1:
                periods.append(fraction[1])
                
        # Find factors using the periods
        factors = set()
        
        for r in periods:
            # If r is even, compute gcd(a^(r/2) ± 1, N)
            if r % 2 == 0:
                x = pow(a, r // 2, N)
                factor1 = math.gcd(x + 1, N)
                factor2 = math.gcd(x - 1, N)
                
                if 1 < factor1 < N:
                    factors.add(factor1)
                if 1 < factor2 < N:
                    factors.add(factor2)
                    
        return {
            "factors": list(factors),
            "periods": periods,
            "measurements": results
        }
        
    def _continued_fraction_expansion(self, numerator: int, denominator: int) -> Tuple[int, int]:
        """
        Find the continued fraction expansion of numerator/denominator.
        Returns the closest convergent as (numerator, denominator).
        """
        import math
        
        a = numerator
        b = denominator
        convergents = []
        
        while b:
            convergents.append(a // b)
            a, b = b, a % b
            
        # Calculate the convergents
        n = [1, convergents[0]]
        d = [0, 1]
        
        for i in range(2, len(convergents)):
            n.append(convergents[i-1] * n[i-1] + n[i-2])
            d.append(convergents[i-1] * d[i-1] + d[i-2])
            
        return (n[-1], d[-1])
        
    def simulate_vqe(self, 
                   hamiltonian: np.ndarray, 
                   ansatz: Callable[[QuantumCircuit, List[float]], None],
                   initial_params: List[float],
                   optimizer: Callable[[Callable[[List[float]], float], List[float]], Tuple[List[float], float]],
                   n_qubits: int) -> Dict[str, Any]:
        """
        Simulate the Variational Quantum Eigensolver (VQE) algorithm.
        
        Args:
            hamiltonian: The Hamiltonian matrix whose ground state energy we want to find
            ansatz: Function that applies the ansatz circuit with given parameters
            initial_params: Initial parameters for the ansatz
            optimizer: Classical optimization function
            n_qubits: Number of qubits in the system
            
        Returns:
            Dictionary with optimized parameters and energy
        """
        # Define the objective function (energy)
        def objective(params: List[float]) -> float:
            # Create a circuit with the ansatz
            circuit = QuantumCircuit(n_qubits)
            ansatz(circuit, params)
            
            # Get the state vector
            statevector = circuit.get_statevector()
            
            # Calculate the expectation value ⟨ψ|H|ψ⟩
            energy = np.real(np.vdot(statevector, hamiltonian @ statevector))
            
            return energy
            
        # Run the classical optimizer
        optimal_params, minimal_energy = optimizer(objective, initial_params)
        
        # Create the optimal circuit
        optimal_circuit = QuantumCircuit(n_qubits)
        ansatz(optimal_circuit, optimal_params)
        
        # Get the final state
        optimal_state = optimal_circuit.get_statevector()
        
        return {
            "optimal_params": optimal_params,
            "ground_state_energy": minimal_energy,
            "ground_state": optimal_state
        }

# ============================================================================
# Membrane Module
# ============================================================================

class Membrane:
    """Base class for membrane filtering and validation."""
    
    def __init__(self):
        """Initialize a membrane with default configuration."""
        self.filters = []
        self.transformers = []
        self.validators = []
        self.cache = {}
        self.stats = {"processed": 0, "filtered": 0, "validated": 0, "transformed": 0}
        
    def add_filter(self, filter_func: Callable[[Any], bool], name: str = None):
        """Add a filter function to the membrane."""
        filter_name = name or f"filter_{len(self.filters)}"
        self.filters.append((filter_func, filter_name))
        return self
        
    def add_transformer(self, transformer_func: Callable[[Any], Any], name: str = None):
        """Add a transformer function to the membrane."""
        transformer_name = name or f"transformer_{len(self.transformers)}"
        self.transformers.append((transformer_func, transformer_name))
        return self
        
    def add_validator(self, validator_func: Callable[[Any], Tuple[bool, str]], name: str = None):
        """Add a validator function to the membrane."""
        validator_name = name or f"validator_{len(self.validators)}"
        self.validators.append((validator_func, validator_name))
        return self
        
    def process(self, data: Any) -> Tuple[bool, Any, Dict[str, Any]]:
        """
        Process data through the membrane.
        
        Returns:
            Tuple of (success, processed_data, metadata)
        """
        self.stats["processed"] += 1
        metadata = {"original_type": type(data).__name__, "filters": [], "transformers": [], "validators": []}
        
        # Check cache first
        cache_key = self._get_cache_key(data)
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        # Apply filters
        for filter_func, filter_name in self.filters:
            if not filter_func(data):
                metadata["filters"].append({"name": filter_name, "passed": False})
                self.stats["filtered"] += 1
                
                result = (False, None, metadata)
                self.cache[cache_key] = result
                return result
            metadata["filters"].append({"name": filter_name, "passed": True})
            
        # Apply transformers
        transformed_data = data
        for transformer_func, transformer_name in self.transformers:
            try:
                transformed_data = transformer_func(transformed_data)
                metadata["transformers"].append({
                    "name": transformer_name, 
                    "success": True,
                    "output_type": type(transformed_data).__name__
                })
                self.stats["transformed"] += 1
            except Exception as e:
                metadata["transformers"].append({
                    "name": transformer_name, 
                    "success": False,
                    "error": str(e)
                })
                
                result = (False, None, metadata)
                self.cache[cache_key] = result
                return result
                
        # Apply validators
        for validator_func, validator_name in self.validators:
            try:
                valid, message = validator_func(transformed_data)
                metadata["validators"].append({
                    "name": validator_name,
                    "valid": valid,
                    "message": message
                })
                
                if not valid:
                    self.stats["validated"] += 1
                    
                    result = (False, None, metadata)
                    self.cache[cache_key] = result
                    return result
            except Exception as e:
                metadata["validators"].append({
                    "name": validator_name,
                    "valid": False,
                    "error": str(e)
                })
                
                result = (False, None, metadata)
                self.cache[cache_key] = result
                return result
                
        # All checks passed
        result = (True, transformed_data, metadata)
        self.cache[cache_key] = result
        return result
        
    def _get_cache_key(self, data: Any) -> str:
        """Generate a cache key for the data."""
        try:
            # Try using the data directly as a key
            if isinstance(data, (str, int, float, bool)):
                return f"{type(data).__name__}:{str(data)}"
                
            # For more complex types, use a hash
            if hasattr(data, '__hash__') and callable(data.__hash__):
                return f"hash:{hash(data)}"
                
            # For unhashable types, use a representation
            return f"repr:{repr(data)[:100]}"
        except:
            # Fall back to object ID
            return f"id:{id(data)}"
            
    def clear_cache(self):
        """Clear the membrane's cache."""
        self.cache = {}
        
    def reset_stats(self):
        """Reset the membrane's statistics."""
        self.stats = {"processed": 0, "filtered": 0, "validated": 0, "transformed": 0}
        
    def get_stats(self) -> Dict[str, Any]:
        """Get the membrane's statistics."""
        return self.stats.copy()

class DataMembrane(Membrane):
    """Specialized membrane for data processing."""
    
    def __init__(self, expected_schema: Optional[Dict[str, Any]] = None):
        """Initialize a data membrane with an optional expected schema."""
        super().__init__()
        self.expected_schema = expected_schema
        
        # Add default validators if schema is provided
        if expected_schema:
            self.add_validator(self._schema_validator, "schema_validator")
            
    def _schema_validator(self, data: Any) -> Tuple[bool, str]:
        """Validate data against the expected schema."""
        if self.expected_schema is None:
            return True, "No schema to validate against"
            
        if not isinstance(data, dict):
            return False, f"Expected dictionary, got {type(data).__name__}"
            
        # Check required fields
        for field, field_spec in self.expected_schema.items():
            if field_spec.get("required", False) and field not in data:
                return False, f"Missing required field: {field}"
                
            if field in data:
                # Check type
                expected_type = field_spec.get("type")
                if expected_type and not isinstance(data[field], expected_type):
                    return False, f"Field {field} has incorrect type: {type(data[field]).__name__}, expected {expected_type.__name__}"
                    
                # Check constraints
                constraints = field_spec.get("constraints", {})
                
                # Numeric constraints
                if "min" in constraints and data[field] < constraints["min"]:
                    return False, f"Field {field} value {data[field]} is less than minimum {constraints['min']}"
                if "max" in constraints and data[field] > constraints["max"]:
                    return False, f"Field {field} value {data[field]} is greater than maximum {constraints['max']}"
                    
                # String constraints
                if "pattern" in constraints and not re.match(constraints["pattern"], data[field]):
                    return False, f"Field {field} value {data[field]} does not match pattern {constraints['pattern']}"
                if "max_length" in constraints and len(data[field]) > constraints["max_length"]:
                    return False, f"Field {field} length {len(data[field])} exceeds maximum {constraints['max_length']}"
                if "min_length" in constraints and len(data[field]) < constraints["min_length"]:
                    return False, f"Field {field} length {len(data[field])} is less than minimum {constraints['min_length']}"
                    
                # Enum constraints
                if "enum" in constraints and data[field] not in constraints["enum"]:
                    return False, f"Field {field} value {data[field]} is not in allowed values: {constraints['enum']}"
                    
        return True, "Data validated successfully"
        
    def add_type_filter(self, expected_type: type):
        """Add a filter that checks for a specific type."""
        return self.add_filter(
            lambda data: isinstance(data, expected_type),
            f"type_filter_{expected_type.__name__}"
        )
        
    def add_null_filter(self):
        """Add a filter that rejects null/None values."""
        return self.add_filter(
            lambda data: data is not None,
            "null_filter"
        )
        
    def add_empty_filter(self):
        """Add a filter that rejects empty collections."""
        return self.add_filter(
            lambda data: not (hasattr(data, "__len__") and len(data) == 0),
            "empty_filter"
        )
        
    def add_range_validator(self, min_val: float, max_val: float):
        """Add a validator that checks if numeric values are within a range."""
        def range_validator(data: Any) -> Tuple[bool, str]:
            if not isinstance(data, (int, float)):
                return False, f"Expected numeric type, got {type(data).__name__}"
            if data < min_val:
                return False, f"Value {data} is less than minimum {min_val}"
            if data > max_val:
                return False, f"Value {data} is greater than maximum {max_val}"
            return True, "Value is within range"
            
        return self.add_validator(range_validator, f"range_validator_{min_val}_{max_val}")
        
    def add_length_validator(self, min_length: int, max_length: int):
        """Add a validator that checks if collections have proper length."""
        def length_validator(data: Any) -> Tuple[bool, str]:
            if not hasattr(data, "__len__"):
                return False, f"Object of type {type(data).__name__} has no length"
            length = len(data)
            if length < min_length:
                return False, f"Length {length} is less than minimum {min_length}"
            if length > max_length:
                return False, f"Length {length} exceeds maximum {max_length}"
            return True, "Length is within range"
            
        return self.add_validator(length_validator, f"length_validator_{min_length}_{max_length}")
        
    def add_regex_validator(self, pattern: str):
        """Add a validator that checks if strings match a regex pattern."""
        import re
        compiled_pattern = re.compile(pattern)
        
        def regex_validator(data: Any) -> Tuple[bool, str]:
            if not isinstance(data, str):
                return False, f"Expected string, got {type(data).__name__}"
            if not compiled_pattern.match(data):
                return False, f"String '{data}' does not match pattern '{pattern}'"
            return True, "String matches pattern"
            
        return self.add_validator(regex_validator, f"regex_validator_{pattern}")
        
    def add_json_transformer(self):
        """Add a transformer that parses JSON strings."""
        import json
        
        def json_transformer(data: Any) -> Any:
            if isinstance(data, str):
                return json.loads(data)
            return data
            
        return self.add_transformer(json_transformer, "json_transformer")
        
    def add_string_transformer(self):
        """Add a transformer that converts data to strings."""
        return self.add_transformer(
            lambda data: str(data),
            "string_transformer"
        )
        
    def add_numpy_transformer(self):
        """Add a transformer that converts lists to numpy arrays."""
        def numpy_transformer(data: Any) -> Any:
            if isinstance(data, (list, tuple)) and all(isinstance(x, (int, float)) for x in data):
                return np.array(data)
            return data
            
        return self.add_transformer(numpy_transformer, "numpy_transformer")

class BinaryMembrane(Membrane):
    """Specialized membrane for binary data processing."""
    
    def __init__(self, expected_magic_numbers: Optional[List[bytes]] = None):
        """Initialize a binary membrane with optional expected magic numbers."""
        super().__init__()
        self.expected_magic_numbers = expected_magic_numbers
        
        # Add default validators if magic numbers are provided
        if expected_magic_numbers:
            self.add_validator(self._magic_number_validator, "magic_number_validator")
            
    def _magic_number_validator(self, data: bytes) -> Tuple[bool, str]:
        """Validate binary data against expected magic numbers."""
        if not self.expected_magic_numbers:
            return True, "No magic numbers to validate against"
            
        if not isinstance(data, bytes):
            return False, f"Expected bytes, got {type(data).__name__}"
            
        if not data:
            return False, "Empty data"
            
        # Check if data starts with any of the expected magic numbers
        for magic in self.expected_magic_numbers:
            if data.startswith(magic):
                return True, f"Magic number matched: {magic.hex()}"
                
        return False, f"No matching magic number found in {data[:20].hex()}"
        
    def add_size_filter(self, min_size: int, max_size: Optional[int] = None):
        """Add a filter that checks if binary data size is within range."""
        def size_filter(data: Any) -> bool:
            if not isinstance(data, bytes):
                return False
            if len(data) < min_size:
                return False
            if max_size is not None and len(data) > max_size:
                return False
            return True
            
        return self.add_filter(size_filter, f"size_filter_{min_size}_{max_size}")
        
    def add_checksum_validator(self, checksum_func: Callable[[bytes], bytes], expected_checksum: bytes):
        """Add a validator that checks if data matches an expected checksum."""
        def checksum_validator(data: bytes) -> Tuple[bool, str]:
            if not isinstance(data, bytes):
                return False, f"Expected bytes, got {type(data).__name__}"
                
            computed_checksum = checksum_func(data)
            if computed_checksum != expected_checksum:
                return False, f"Checksum mismatch: got {computed_checksum.hex()}, expected {expected_checksum.hex()}"
                
            return True, "Checksum validated"
            
        return self.add_validator(checksum_validator, "checksum_validator")
        
    def add_decompression_transformer(self, algorithm: str = 'zlib'):
        """Add a transformer that decompresses binary data."""
        import zlib
        
        def decompress_transformer(data: bytes) -> bytes:
            if not isinstance(data, bytes):
                return data
                
            if algorithm == 'zlib':
                return zlib.decompress(data)
            elif algorithm == 'gzip':
                return zlib.decompress(data, 16 + zlib.MAX_WBITS)
            else:
                raise ValueError(f"Unsupported compression algorithm: {algorithm}")
                
        return self.add_transformer(decompress_transformer, f"decompress_{algorithm}_transformer")
        
    def add_encryption_transformer(self, key: bytes, algorithm: str = 'xor'):
        """Add a transformer that decrypts binary data."""
        def decrypt_xor(data: bytes, key: bytes) -> bytes:
            """Simple XOR decryption."""
            result = bytearray(len(data))
            for i, b in enumerate(data):
                result[i] = b ^ key[i % len(key)]
            return bytes(result)
            
        def decrypt_transformer(data: bytes) -> bytes:
            if not isinstance(data, bytes):
                return data
                
            if algorithm == 'xor':
                return decrypt_xor(data, key)
            else:
                raise ValueError(f"Unsupported encryption algorithm: {algorithm}")
                
        return self.add_transformer(decrypt_transformer, f"decrypt_{algorithm}_transformer")
        
    def add_hex_transformer(self):
        """Add a transformer that converts binary data to hex strings."""
        return self.add_transformer(
            lambda data: data.hex() if isinstance(data, bytes) else data,
            "hex_transformer"
        )
        
    def add_hash_transformer(self, algorithm: str = 'sha256'):
        """Add a transformer that hashes binary data."""
        def hash_transformer(data: Any) -> str:
            if isinstance(data, bytes):
                if algorithm == 'md5':
                    return hashlib.md5(data).hexdigest()
                elif algorithm == 'sha1':
                    return hashlib.sha1(data).hexdigest()
                elif algorithm == 'sha256':
                    return hashlib.sha256(data).hexdigest()
                else:
                    raise ValueError(f"Unsupported hash algorithm: {algorithm}")
            return data
            
        return self.add_transformer(hash_transformer, f"hash_{algorithm}_transformer")

# ============================================================================
# Node Network Module
# ============================================================================

class NodeType(Enum):
    """Types of nodes in the network."""
    INPUT = auto()
    PROCESSING = auto()
    OUTPUT = auto()
    SUPERNODE = auto()
    STORAGE = auto()
    FILTERING = auto()
    MEMORY = auto()
    VISUALIZATION = auto()
    QUANTUM = auto()
    CUSTOM = auto()

class MessageType(Enum):
    """Types of messages that can be passed between nodes."""
    DATA = auto()
    COMMAND = auto()
    STATUS = auto()
    ERROR = auto()
    INFO = auto()
    RESULT = auto()
    REQUEST = auto()
    RESPONSE = auto()
    EVENT = auto()
    CUSTOM = auto()

@dataclass
class Message:
    """Message passed between nodes."""tion."""
    msg_type: MessageType
    content: Any
    sender_id: str
    timestamp: float = field(default_factory=time.time)
    message_id: str = field(default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest())
    priority: int = 0r, Any] = field(default_factory=dict)
    ttl: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Base class for all nodes in the network."""
class Node(ABC):
    """Base class for all nodes in the network."""deType):
        """Initialize a node with an ID and type."""
    def __init__(self, node_id: str, node_type: NodeType):
        """Initialize a node."""
        self.node_id = node_id # Map of node_id -> Node
        self.node_type = node_type
        self.connections = {}
        self.active = True
        self.state = {}
        self.message_queue = [] Membrane()  # Default membrane for filtering
        
    def connect(self, node: 'Node', bidirectional: bool = True) -> 'Node':Node':
        """Connect to another node.""""""Connect this node to another node."""
        self.connections[node.node_id] = node
        if bidirectional:
            node.connections[self.node_id] = selfself
        return self
        
    def disconnect(self, node_id: str, bidirectional: bool = True) -> 'Node':self, node_id: str, bidirectional: bool = True) -> 'Node':
        """Disconnect from a node.""""""Disconnect from another node."""
        if node_id in self.connections:
            node = self.connections[node_id]]
            del self.connections[node_id]d]
            if bidirectional and self.node_id in node.connections:
                del node.connections[self.node_id]e_id in node.connections:
        return self    del node.connections[self.node_id]
        
    def send(self, message: Message, target_id: str) -> bool:
        """Send a message to a specific node."""
        if target_id not in self.connections:message: Message, target_id: str) -> bool:
            logger.warning(f"Node {self.node_id} cannot send to unknown node {target_id}")"""Send a message to a specific node."""
            return False
             cannot send to unknown node {target_id}")
        target = self.connections[target_id]
        return target.receive(message)
        nnections[target_id]
    def broadcast(self, message: Message) -> int:rn target.receive(message)
        """Broadcast a message to all connected nodes."""
        success_count = 0ge) -> int:
        """Broadcast a message to all connected nodes."""
        for node_id in self.connections:
            if self.send(message, node_id):
                success_count += 1lf.connections:
                    if self.send(message, node_id):
        return success_count
        
    def receive(self, message: Message) -> bool:
        """Receive a message and add it to the queue."""
        if not self.active:e: Message) -> bool:
            logger.warning(f"Node {self.node_id} is inactive, rejecting message {message.message_id}")"""Receive a message and add it to the queue."""
            return False
            tive, rejecting message {message.message_id}")
        # Check if we've already processed this message
        if message.message_id in self.processed_messages:
            logger.warning(f"Node {self.node_id} already processed message {message.message_id}") already processed this message
            return Falseessage.message_id in self.processed_messages:
            y processed message {message.message_id}")
        # Filter message through membrane
        success, processed_message, metadata = self.membrane.process(message)
         through membrane
        if not success:ess, processed_message, metadata = self.membrane.process(message)
            logger.warning(f"Node {self.node_id} membrane rejected message {message.message_id}: {metadata}")
            return False
                logger.warning(f"Node {self.node_id} membrane rejected message {message.message_id}: {metadata}")
        # Add to queue based on prioritye
        heapq.heappush(self.message_queue, (-message.priority, message))
        return Trueased on priority
        q.heappush(self.message_queue, (-message.priority, message))
    def process_next(self) -> Optional[Message]:
        """Process the next message in the queue."""
        if not self.message_queue:t(self) -> Optional[Message]:
            return None"""Process the next message in the queue."""
            
        _, message = heapq.heappop(self.message_queue)
        
        # Mark as processedapq.heappop(self.message_queue)
        self.processed_messages[message.message_id] = time.time()
        
        # Process the messageself.processed_messages[message.message_id] = time.time()
        self._process_message(message)
        
        return messageself._process_message(message)
        
    @abstractmethod
    def _process_message(self, message: Message) -> Any:
        """Process a message (to be implemented by subclasses)."""
        pass_process_message(self, message: Message) -> Any:
        a message (to be implemented by subclasses)."""
    def process_all(self) -> int:
        """Process all messages in the queue."""
        count = 0ess_all(self) -> int:
        while self.message_queue:"""Process all messages in the queue."""
            self.process_next()
            count += 1
        return countprocess_next()
        
    def create_message(self, msg_type: MessageType, content: Any, priority: int = 0) -> Message:
        """Create a new message originating from this node."""
        return Message(ge(self, msg_type: MessageType, content: Any, priority: int = 0) -> Message:
            msg_type=msg_type,"""Create a new message originating from this node."""
            content=content,
            sender_id=self.node_id,
            priority=prioritytent,
        )e_id,
        y
    def activate(self):
        """Activate the node."""
        self.active = Truectivate(self):
        """Activate the node."""
    def deactivate(self):rue
        """Deactivate the node."""
        self.active = False
        """Deactivate the node."""
    def save_state(self, filepath: str):se
        """Save the node's state to a file."""
        with open(filepath, 'wb') as f:lepath: str):
            pickle.dump({"""Save the node's state to a file."""
                'node_id': self.node_id,
                'node_type': self.node_type,
                'state': self.state,,
                'active': self.activee': self.node_type,
            }, f)
            
    def load_state(self, filepath: str):
        """Load the node's state from a file."""
        with open(filepath, 'rb') as f:e(self, filepath: str):
            data = pickle.load(f)oad the node's state from a file."""
            
        if data['node_id'] != self.node_id:
            logger.warning(f"Loading state from node {data['node_id']} into node {self.node_id}")
            f.node_id:
        self.state = data['state']logger.warning(f"Loading state from node {data['node_id']} into node {self.node_id}")
        self.active = data['active']
        
    def get_connections(self) -> List[str]:.active = data['active']
        """Get a list of all connected node IDs."""
        return list(self.connections.keys())t[str]:
        """Get a list of all connected node IDs."""
    def get_stats(self) -> Dict[str, Any]:)
        """Get statistics about this node."""
        return {
            "node_id": self.node_id,"""Get statistics about this node."""
            "node_type": self.node_type.name,
            "connections": len(self.connections),
            "queue_size": len(self.message_queue),e_type": self.node_type.name,
            "processed_messages": len(self.processed_messages),connections),
            "active": self.active,eue),
            "membrane_stats": self.membrane.get_stats()sed_messages),
        }

class ProcessingNode(Node):
    """Node that processes data with a specific function."""
    cessingNode(Node):
    def __init__(self, node_id: str, process_func: Callable[[Any], Any]):    """Node that processes data with a specific function."""
        """Initialize a processing node with a processing function."""
        super().__init__(node_id, NodeType.PROCESSING)[Any], Any]):
        self.process_func = process_func    """Initialize a processing node with a processing function."""
        self.results = {}
        
    def _process_message(self, message: Message) -> Any:
        """Process a message using the processing function."""
        if message.msg_type == MessageType.DATA:self, message: Message) -> Any:
            try:"""Process a message using the processing function."""
                # Process the data
                result = self.process_func(message.content)
                
                # Store the resultresult = self.process_func(message.content)
                self.results[message.message_id] = result
                
                # Create a result messageself.results[message.message_id] = result
                result_message = self.create_message(
                    MessageType.RESULT,
                    result,result_message = self.create_message(
                    priority=message.priority
                )
                iority
                # Set metadata
                result_message.metadata = {
                    "original_message_id": message.message_id, Set metadata
                    "processing_time": time.time() - message.timestamp,result_message.metadata = {
                    "success": Truemessage_id": message.message_id,
                }.time() - message.timestamp,
                
                # Broadcast the result
                self.broadcast(result_message)
                 Broadcast the result
                return resultself.broadcast(result_message)
            except Exception as e:
                # Create an error message
                error_message = self.create_message(pt Exception as e:
                    MessageType.ERROR,rror message
                    str(e),lf.create_message(
                    priority=message.priority
                )
                riority
                # Set metadata
                error_message.metadata = {
                    "original_message_id": message.message_id, Set metadata
                    "error_type": type(e).__name__,error_message.metadata = {
                    "stack_trace": traceback.format_exc()message_id": message.message_id,
                }__name__,
                
                # Broadcast the error
                self.broadcast(error_message)
                 Broadcast the error
                return Noneself.broadcast(error_message)
        else:
            # For non-data messages, just log and ignore
            logger.info(f"ProcessingNode {self.node_id} received non-data message: {message.msg_type}")
            return Nonemessages, just log and ignore
            ogger.info(f"ProcessingNode {self.node_id} received non-data message: {message.msg_type}")
    def get_result(self, message_id: str) -> Optional[Any]:
        """Get the result of processing a specific message."""
        return self.results.get(message_id), message_id: str) -> Optional[Any]:
        et the result of processing a specific message."""
    def clear_results(self):
        """Clear all stored results."""
        self.results = {}
"""Clear all stored results."""
class SuperNode(Node):
    """A node that contains and manages other nodes."""
    
    def __init__(self, node_id: str):    """A node that contains and manages other nodes."""
        """Initialize a super node."""
        super().__init__(node_id, NodeType.SUPERNODE)
        self.subnodes = {}  # Map of node_id -> Node    """Initialize a super node."""
        self.routing_table = {}  # Map of message types to target node IDseType.SUPERNODE)
        ode_id -> Node
    def add_node(self, node: Node) -> 'SuperNode':es to target node IDs
        """Add a node to this super node."""
        self.subnodes[node.node_id] = node
        return self"""Add a node to this super node."""
        
    def remove_node(self, node_id: str) -> Optional[Node]:
        """Remove a node from this super node."""
        if node_id in self.subnodes:(self, node_id: str) -> Optional[Node]:
            node = self.subnodes[node_id]"""Remove a node from this super node."""
            del self.subnodes[node_id]
            return node
        return Noned]
        
    def add_route(self, msg_type: MessageType, target_id: str) -> 'SuperNode':
        """Add a routing rule for a message type."""
        if target_id not in self.subnodes and target_id not in self.connections:elf, msg_type: MessageType, target_id: str) -> 'SuperNode':
            logger.warning(f"SuperNode {self.node_id} adding route to unknown node {target_id}")"""Add a routing rule for a message type."""
            s:
        if msg_type not in self.routing_table:} adding route to unknown node {target_id}")
            self.routing_table[msg_type] = []
            
        self.routing_table[msg_type].append(target_id)self.routing_table[msg_type] = []
        return self
        arget_id)
    def _process_message(self, message: Message) -> Any:rn self
        """Process a message by routing it to the appropriate subnode."""
        if message.msg_type in self.routing_table:ssage(self, message: Message) -> Any:
            # Route to all appropriate targets"""Process a message by routing it to the appropriate subnode."""
            targets = self.routing_table[message.msg_type]
            results = []
            sg_type]
            for target_id in targets:
                if target_id in self.subnodes:
                    # Internal routingd in targets:
                    success = self.subnodes[target_id].receive(message)    if target_id in self.subnodes:
                    if success:g
                        # Process immediatelyrget_id].receive(message)
                        result = self.subnodes[target_id].process_next()
                        results.append(result)
                elif target_id in self.connections:= self.subnodes[target_id].process_next()
                    # External routing)
                    success = self.send(message, target_id)
                    if success:
                        results.append(True)rget_id)
                else:
                    logger.warning(f"SuperNode {self.node_id} cannot route message to unknown node {target_id}")
                    
            return resultsde {self.node_id} cannot route message to unknown node {target_id}")
        else:
            # Process in all subnodes
            for node in self.subnodes.values():
                node.receive(message)ll subnodes
                node.process_next()or node in self.subnodes.values():
                
            # Also broadcast to external connections
            self.broadcast(message)
            ernal connections
            return None.broadcast(message)
            
    def process_all_subnodes(self) -> Dict[str, int]:
        """Process all messages in all subnodes."""
        results = {}nodes(self) -> Dict[str, int]:
        rocess all messages in all subnodes."""
        for node_id, node in self.subnodes.items():
            count = node.process_all()
            results[node_id] = count node in self.subnodes.items():
                count = node.process_all()
        return results
        
    def activate_all(self):
        """Activate all subnodes."""
        for node in self.subnodes.values():elf):
            node.activate()"""Activate all subnodes."""
            bnodes.values():
    def deactivate_all(self):
        """Deactivate all subnodes."""
        for node in self.subnodes.values():):
            node.deactivate()eactivate all subnodes."""
            odes.values():
    def get_subnode(self, node_id: str) -> Optional[Node]:
        """Get a specific subnode by ID."""
        return self.subnodes.get(node_id)e_id: str) -> Optional[Node]:
        et a specific subnode by ID."""
    def get_all_subnodes(self) -> Dict[str, Node]:
        """Get all subnodes."""
        return self.subnodes.copy()r, Node]:
        """Get all subnodes."""
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about this supernode and its subnodes."""
        stats = super().get_stats(), Any]:
        """Get statistics about this supernode and its subnodes."""
        # Add supernode-specific stats
        stats["subnodes"] = len(self.subnodes)
        stats["routes"] = {msg_type.name: targets for msg_type, targets in self.routing_table.items()}ats
        stats["subnodes"] = len(self.subnodes)
        # Add subnode statsme: targets for msg_type, targets in self.routing_table.items()}
        subnode_stats = {}
        for node_id, node in self.subnodes.items():
            subnode_stats[node_id] = node.get_stats()subnode_stats = {}
            n self.subnodes.items():
        stats["subnode_stats"] = subnode_statsnode_id] = node.get_stats()
        
        return stats

class InputNode(Node):
    """Node that serves as an input point for data."""
    ):
    def __init__(self, node_id: str, input_type: type):    """Node that serves as an input point for data."""
        """Initialize an input node with an expected input type."""
        super().__init__(node_id, NodeType.INPUT):
        self.input_type = input_type    """Initialize an input node with an expected input type."""
        
        # Add a type filter to the membrane
        self.membrane.add_filter(
            lambda data: isinstance(data.content, self.input_type),embrane
            f"input_type_filter_{self.input_type.__name__}"self.membrane.add_filter(
        )ntent, self.input_type),
        self.input_type.__name__}"
    def _process_message(self, message: Message) -> Any:
        """Process input messages and forward them."""
        # Simply forward all messages to all connected nodesprocess_message(self, message: Message) -> Any:
        self.broadcast(message)"""Process input messages and forward them."""
        return message.contentodes
        
    def input_data(self, data: Any, priority: int = 0) -> bool:
        """Input data to the node and broadcast it."""
        if not isinstance(data, self.input_type): Any, priority: int = 0) -> bool:
            logger.warning(f"InputNode {self.node_id} expected {self.input_type.__name__}, got {type(data).__name__}")"""Input data to the node and broadcast it."""
            return False
            expected {self.input_type.__name__}, got {type(data).__name__}")
        # Create a data message
        message = self.create_message(MessageType.DATA, data, priority)
        message
        # Add to the queue and process immediatelyage = self.create_message(MessageType.DATA, data, priority)
        heapq.heappush(self.message_queue, (-priority, message))
        self.process_next()
        heapq.heappush(self.message_queue, (-priority, message))
        return True

class OutputNode(Node):
    """Node that serves as an output point for results."""
    de):
    def __init__(self, node_id: str, output_handler: Callable[[Any], None] = None):    """Node that serves as an output point for results."""
        """Initialize an output node with an optional output handler."""
        super().__init__(node_id, NodeType.OUTPUT)ble[[Any], None] = None):
        self.output_handler = output_handler or (lambda x: None)    """Initialize an output node with an optional output handler."""
        self.outputs = []
        
    def _process_message(self, message: Message) -> Any:
        """Process output messages."""
        # Store the outputself, message: Message) -> Any:
        self.outputs.append((message, time.time()))"""Process output messages."""
        
        # Call the output handlertime.time()))
        self.output_handler(message.content)
        
        return message.contentself.output_handler(message.content)
        
    def get_outputs(self, n: Optional[int] = None) -> List[Tuple[Message, float]]:
        """Get the latest n outputs with timestamps."""
        if n is None:ptional[int] = None) -> List[Tuple[Message, float]]:
            return self.outputs.copy()"""Get the latest n outputs with timestamps."""
        return self.outputs[-n:]
        
    def clear_outputs(self):utputs[-n:]
        """Clear all stored outputs."""
        self.outputs = []
        """Clear all stored outputs."""
    def set_output_handler(self, handler: Callable[[Any], None]):
        """Set a new output handler."""
        self.output_handler = handlerr(self, handler: Callable[[Any], None]):
"""Set a new output handler."""
class StorageNode(Node):
    """Node that stores data for later retrieval."""
    
    def __init__(self, node_id: str, capacity: Optional[int] = None):    """Node that stores data for later retrieval."""
        """Initialize a storage node with an optional capacity limit."""
        super().__init__(node_id, NodeType.STORAGE)nal[int] = None):
        self.capacity = capacity    """Initialize a storage node with an optional capacity limit."""
        self.storage = {}  # Key-value store
        self.timestamps = {}  # Timestamp for each key
        self.access_counts = {}  # Access count for each key
        Timestamp for each key
    def _process_message(self, message: Message) -> Any:unt for each key
        """Process storage-related messages."""
        if message.msg_type == MessageType.DATA:
            # Check if the message has a key in its metadata"""Process storage-related messages."""
            key = message.metadata.get("storage_key")
             its metadata
            if key is None:key")
                # Generate a key based on the message ID
                key = f"msg_{message.message_id}"
                    # Generate a key based on the message ID
            # Store the data_{message.message_id}"
            success = self.store(key, message.content)
            
            # Create a status messageess = self.store(key, message.content)
            status_message = self.create_message(
                MessageType.STATUS,
                {"success": success, "key": key},status_message = self.create_message(
                priority=message.priority
            )
            iority
            # Broadcast the status
            self.broadcast(status_message)
             Broadcast the status
            return successself.broadcast(status_message)
        elif message.msg_type == MessageType.REQUEST:
            # Handle data retrieval requests
            request = message.content message.msg_type == MessageType.REQUEST:
            retrieval requests
            if isinstance(request, dict) and "key" in request:
                key = request["key"]
                data = self.retrieve(key)ct) and "key" in request:
                    key = request["key"]
                # Create a response message
                response_message = self.create_message(
                    MessageType.RESPONSE,ge
                    {"key": key, "data": data},response_message = self.create_message(
                    priority=message.priority
                )
                rity
                # Send the response back to the sender
                self.send(response_message, message.sender_id)
                 Send the response back to the sender
                return dataself.send(response_message, message.sender_id)
                
        return None
        
    def store(self, key: str, data: Any) -> bool:
        """Store data under a key."""
        # Check capacity key: str, data: Any) -> bool:
        if self.capacity is not None and len(self.storage) >= self.capacity and key not in self.storage:"""Store data under a key."""
            # Implement LRU eviction
            if self.timestamps:and len(self.storage) >= self.capacity and key not in self.storage:
                # Find the least recently used keyLRU eviction
                lru_key = min(self.timestamps, key=self.timestamps.get)
                # Remove itently used key
                del self.storage[lru_key]elf.timestamps, key=self.timestamps.get)
                del self.timestamps[lru_key]
                del self.access_counts[lru_key]
                mestamps[lru_key]
        # Store the datau_key]
        self.storage[key] = data
        self.timestamps[key] = time.time()
        self.access_counts[key] = self.access_counts.get(key, 0)rage[key] = data
        key] = time.time()
        return True= self.access_counts.get(key, 0)
        
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve data by key."""
        if key in self.storage:lf, key: str) -> Optional[Any]:
            # Update timestamp and access count"""Retrieve data by key."""
            self.timestamps[key] = time.time()
            self.access_counts[key] = self.access_counts.get(key, 0) + 1access count
            ] = time.time()
            return self.storage[key]ss_counts.get(key, 0) + 1
            
        return None
        
    def delete(self, key: str) -> bool:
        """Delete data by key."""
        if key in self.storage:, key: str) -> bool:
            del self.storage[key]"""Delete data by key."""
            del self.timestamps[key]
            del self.access_counts[key]
            return True[key]
            s[key]
        return False
        
    def clear(self):
        """Clear all stored data."""
        self.storage = {}
        self.timestamps = {}"""Clear all stored data."""
        self.access_counts = {} = {}
        
    def list_keys(self) -> List[str]:s = {}
        """List all stored keys."""
        return list(self.storage.keys())[str]:
        """List all stored keys."""
    def get_stats(self) -> Dict[str, Any]:())
        """Get storage statistics."""
        stats = super().get_stats()]:
        """Get storage statistics."""
        # Add storage-specific stats
        stats["storage_size"] = len(self.storage)
        stats["storage_capacity"] = self.capacitys
        stats["storage_size"] = len(self.storage)
        if self.storage:self.capacity
            stats["oldest_key"] = min(self.timestamps, key=self.timestamps.get)
            stats["newest_key"] = max(self.timestamps, key=self.timestamps.get)
            stats["most_accessed_key"] = max(self.access_counts, key=self.access_counts.get)    stats["oldest_key"] = min(self.timestamps, key=self.timestamps.get)
            stats["most_access_count"] = max(self.access_counts.values()) if self.access_counts else 0t_key"] = max(self.timestamps, key=self.timestamps.get)
            s_counts.get)
        return statslf.access_counts else 0

class FilteringNode(Node):
    """Node that filters messages based on criteria."""
    Node):
    def __init__(self, node_id: str, filter_func: Callable[[Message], bool]):    """Node that filters messages based on criteria."""
        """Initialize a filtering node with a filter function."""
        super().__init__(node_id, NodeType.FILTERING)ble[[Message], bool]):
        self.filter_func = filter_func    """Initialize a filtering node with a filter function."""
        self.filtered_count = 0
        self.passed_count = 0
        
    def _process_message(self, message: Message) -> Any:
        """Process messages by filtering them."""
        # Apply the filter, message: Message) -> Any:
        if self.filter_func(message):"""Process messages by filtering them."""
            # Message passed the filter
            self.passed_count += 1
            ed the filter
            # Forward to all connected nodes
            self.broadcast(message)
            cted nodes
            return Trueself.broadcast(message)
        else:
            # Message didn't pass the filter
            self.filtered_count += 1:
            idn't pass the filter
            # Create a status messageelf.filtered_count += 1
            status_message = self.create_message(
                MessageType.STATUS,e
                {"filtered": True, "message_id": message.message_id},status_message = self.create_message(
                priority=message.priority
            )message.message_id},
            iority
            # Send status message back to sender
            self.send(status_message, message.sender_id)
             Send status message back to sender
            return Falseself.send(status_message, message.sender_id)
            
    def get_stats(self) -> Dict[str, Any]:
        """Get filtering statistics."""
        stats = super().get_stats()-> Dict[str, Any]:
        et filtering statistics."""
        # Add filtering-specific stats
        stats["filtered_count"] = self.filtered_count
        stats["passed_count"] = self.passed_countats
        stats["pass_rate"] = self.passed_count / (self.passed_count + self.filtered_count) if (self.passed_count + self.filtered_count) > 0 else 0stats["filtered_count"] = self.filtered_count
        assed_count
        return statsf.passed_count + self.filtered_count) if (self.passed_count + self.filtered_count) > 0 else 0

class VisualizationNode(Node):
    """Node that creates visualizations of data."""
    ode(Node):
    def __init__(self, node_id: str, visualize_func: Callable[[Any], Any]):    """Node that creates visualizations of data."""
        """Initialize a visualization node with a visualization function."""
        super().__init__(node_id, NodeType.VISUALIZATION): Callable[[Any], Any]):
        self.visualize_func = visualize_func    """Initialize a visualization node with a visualization function."""
        self.visualizations = {}
        
    def _process_message(self, message: Message) -> Any:
        """Process messages by creating visualizations."""
        if message.msg_type == MessageType.DATA:essage: Message) -> Any:
            try:"""Process messages by creating visualizations."""
                # Create visualization
                visualization = self.visualize_func(message.content)
                
                # Store the visualizationvisualization = self.visualize_func(message.content)
                self.visualizations[message.message_id] = visualization
                
                # Create a result messageself.visualizations[message.message_id] = visualization
                result_message = self.create_message(
                    MessageType.RESULT,
                    visualization,result_message = self.create_message(
                    priority=message.priority
                )
                iority
                # Set metadata
                result_message.metadata = {
                    "original_message_id": message.message_id, Set metadata
                    "visualization_type": type(visualization).__name__result_message.metadata = {
                }message_id": message.message_id,
                ype(visualization).__name__
                # Broadcast the result
                self.broadcast(result_message)
                 Broadcast the result
                return visualizationself.broadcast(result_message)
            except Exception as e:
                # Create an error message
                error_message = self.create_message(pt Exception as e:
                    MessageType.ERROR,ssage
                    str(e),lf.create_message(
                    priority=message.priority
                )
                riority
                # Set metadata
                error_message.metadata = {
                    "original_message_id": message.message_id, Set metadata
                    "error_type": type(e).__name__,error_message.metadata = {
                    "stack_trace": traceback.format_exc()message_id": message.message_id,
                }__name__,
                
                # Broadcast the error
                self.broadcast(error_message)
                 Broadcast the error
                return Noneself.broadcast(error_message)
                
        return None
        
    def get_visualization(self, message_id: str) -> Optional[Any]:
        """Get a visualization by message ID."""
        return self.visualizations.get(message_id)zation(self, message_id: str) -> Optional[Any]:
        """Get a visualization by message ID."""
    def clear_visualizations(self):
        """Clear all stored visualizations."""
        self.visualizations = {}
"""Clear all stored visualizations."""
class QuantumNode(Node):
    """Node that performs quantum operations."""
    
    def __init__(self, node_id: str, simulator: QuantumSimulator):    """Node that performs quantum operations."""
        """Initialize a quantum node with a quantum simulator."""
        super().__init__(node_id, NodeType.QUANTUM)QuantumSimulator):
        self.simulator = simulator    """Initialize a quantum node with a quantum simulator."""
        self.circuits = {}
        self.results = {}
        
    def _process_message(self, message: Message) -> Any:
        """Process messages with quantum operations."""
        if message.msg_type == MessageType.COMMAND:self, message: Message) -> Any:
            command = message.content"""Process messages with quantum operations."""
            
            if not isinstance(command, dict):
                logger.warning(f"QuantumNode {self.node_id} expected dict command, got {type(command).__name__}")
                return None, dict):
                    logger.warning(f"QuantumNode {self.node_id} expected dict command, got {type(command).__name__}")
            # Handle different quantum commands
            if "operation" in command:
                operation = command["operation"]ent quantum commands
                operation" in command:
                if operation == "create_circuit":]
                    # Create a new quantum circuit
                    n_qubits = command.get("n_qubits", 1):
                    circuit_id = command.get("circuit_id", f"circuit_{message.message_id}")    # Create a new quantum circuit
                    its", 1)
                    circuit = self.simulator.create_circuit(n_qubits)uit_id", f"circuit_{message.message_id}")
                    self.circuits[circuit_id] = circuit
                    
                    # Create a status messageself.circuits[circuit_id] = circuit
                    status_message = self.create_message(
                        MessageType.STATUS,
                        {"circuit_created": True, "circuit_id": circuit_id, "n_qubits": n_qubits},status_message = self.create_message(
                        priority=message.priority
                    )t_id": circuit_id, "n_qubits": n_qubits},
                    iority
                    # Broadcast the status
                    self.broadcast(status_message)
                     Broadcast the status
                    return circuitself.broadcast(status_message)
                    
                elif operation == "add_gate":
                    # Add a gate to a circuit
                    circuit_id = command.get("circuit_id")"add_gate":
                    gate = command.get("gate")# Add a gate to a circuit
                    targets = command.get("targets", [])"circuit_id")
                    params = command.get("params"))
                    
                    if circuit_id not in self.circuits:ms")
                        logger.warning(f"QuantumNode {self.node_id} unknown circuit {circuit_id}")
                        return Noneuits:
                            logger.warning(f"QuantumNode {self.node_id} unknown circuit {circuit_id}")
                    circuit = self.circuits[circuit_id]
                    circuit.add_gate(gate, targets, params)
                    circuits[circuit_id]
                    return Trueuit.add_gate(gate, targets, params)
                    
                elif operation == "run_circuit":
                    # Run a quantum circuit
                    circuit_id = command.get("circuit_id")== "run_circuit":
                    shots = command.get("shots", 1024)# Run a quantum circuit
                    rcuit_id")
                    if circuit_id not in self.circuits:ots", 1024)
                        logger.warning(f"QuantumNode {self.node_id} unknown circuit {circuit_id}")
                        return None:
                            logger.warning(f"QuantumNode {self.node_id} unknown circuit {circuit_id}")
                    circuit = self.circuits[circuit_id]
                    results = self.simulator.run_circuit(circuit, shots)
                    circuits[circuit_id]
                    # Store the resultslts = self.simulator.run_circuit(circuit, shots)
                    result_id = command.get("result_id", f"result_{message.message_id}")
                    self.results[result_id] = results
                    result_id = command.get("result_id", f"result_{message.message_id}")
                    # Create a result message_id] = results
                    result_message = self.create_message(
                        MessageType.RESULT,
                        {"result_id": result_id, "results": results},result_message = self.create_message(
                        priority=message.priority
                    )": results},
                    iority
                    # Broadcast the result
                    self.broadcast(result_message)
                     Broadcast the result
                    return resultsself.broadcast(result_message)
                    
                elif operation == "qft":
                    # Create a Quantum Fourier Transform circuit
                    n_qubits = command.get("n_qubits", 3)"qft":
                    circuit_id = command.get("circuit_id", f"qft_{message.message_id}")# Create a Quantum Fourier Transform circuit
                    et("n_qubits", 3)
                    circuit = self.simulator.quantum_fourier_transform(n_qubits)_{message.message_id}")
                    self.circuits[circuit_id] = circuit
                    
                    # Create a status messageself.circuits[circuit_id] = circuit
                    status_message = self.create_message(
                        MessageType.STATUS,
                        {"qft_created": True, "circuit_id": circuit_id, "n_qubits": n_qubits},status_message = self.create_message(
                        priority=message.priority
                    )": circuit_id, "n_qubits": n_qubits},
                    iority
                    # Broadcast the status
                    self.broadcast(status_message)
                     Broadcast the status
                    return circuitself.broadcast(status_message)
                    
                elif operation == "grover":
                    # Run Grover's algorithm
                    n_qubits = command.get("n_qubits", 3)"grover":
                    marked_states = command.get("marked_states", [])# Run Grover's algorithm
                    iterations = command.get("iterations")"n_qubits", 3)
                    circuit_id = command.get("circuit_id", f"grover_{message.message_id}")get("marked_states", [])
                    )
                    # Create oracle function{message.message_id}")
                    def oracle_function(state: int) -> bool:
                        return state in marked_states
                        def oracle_function(state: int) -> bool:
                    circuit = self.simulator.simulate_grover(n_qubits, oracle_function, iterations)ed_states
                    self.circuits[circuit_id] = circuit
                    _grover(n_qubits, oracle_function, iterations)
                    # Run the circuit.circuits[circuit_id] = circuit
                    shots = command.get("shots", 1024)
                    results = self.simulator.run_circuit(circuit, shots)
                    shots = command.get("shots", 1024)
                    # Store the resultsmulator.run_circuit(circuit, shots)
                    result_id = command.get("result_id", f"result_{message.message_id}")
                    self.results[result_id] = results
                    result_id = command.get("result_id", f"result_{message.message_id}")
                    # Create a result message_id] = results
                    result_message = self.create_message(
                        MessageType.RESULT,
                        {"result_id": result_id, "results": results, "circuit_id": circuit_id},result_message = self.create_message(
                        priority=message.priority
                    )": results, "circuit_id": circuit_id},
                    iority
                    # Broadcast the result
                    self.broadcast(result_message)
                     Broadcast the result
                    return resultsself.broadcast(result_message)
                    
                elif operation == "phase_estimation":
                    # Run Quantum Phase Estimation
                    n_counting_qubits = command.get("n_counting_qubits", 3)"phase_estimation":
                    unitary_matrix = command.get("unitary_matrix")# Run Quantum Phase Estimation
                    circuit_id = command.get("circuit_id", f"qpe_{message.message_id}")n_counting_qubits", 3)
                    unitary_matrix")
                    if unitary_matrix is None:essage_id}")
                        logger.warning(f"QuantumNode {self.node_id} missing unitary_matrix for phase_estimation")
                        return None
                            logger.warning(f"QuantumNode {self.node_id} missing unitary_matrix for phase_estimation")
                    # Convert to numpy array if needed
                    if not isinstance(unitary_matrix, np.ndarray):
                        unitary_matrix = np.array(unitary_matrix, dtype=np.complex128)mpy array if needed
                        ot isinstance(unitary_matrix, np.ndarray):
                    circuit = self.simulator.simulate_quantum_phase_estimation(n_counting_qubits, unitary_matrix)ary_matrix, dtype=np.complex128)
                    self.circuits[circuit_id] = circuit
                    ing_qubits, unitary_matrix)
                    # Run the circuit.circuits[circuit_id] = circuit
                    shots = command.get("shots", 1024)
                    results = self.simulator.run_circuit(circuit, shots)
                    shots = command.get("shots", 1024)
                    # Store the resultsmulator.run_circuit(circuit, shots)
                    result_id = command.get("result_id", f"result_{message.message_id}")
                    self.results[result_id] = results
                    result_id = command.get("result_id", f"result_{message.message_id}")
                    # Create a result message_id] = results
                    result_message = self.create_message(
                        MessageType.RESULT,
                        {"result_id": result_id, "results": results, "circuit_id": circuit_id},result_message = self.create_message(
                        priority=message.priority
                    )": results, "circuit_id": circuit_id},
                    iority
                    # Broadcast the result
                    self.broadcast(result_message)
                     Broadcast the result
                    return resultsself.broadcast(result_message)
                    
        return None
        
    def get_circuit(self, circuit_id: str) -> Optional[QuantumCircuit]:
        """Get a quantum circuit by ID."""
        return self.circuits.get(circuit_id)(self, circuit_id: str) -> Optional[QuantumCircuit]:
        """Get a quantum circuit by ID."""
    def get_result(self, result_id: str) -> Optional[Dict[str, int]]:
        """Get a quantum result by ID."""
        return self.results.get(result_id)Optional[Dict[str, int]]:
        """Get a quantum result by ID."""
    def clear_circuits(self):
        """Clear all stored circuits."""
        self.circuits = {}
        """Clear all stored circuits."""
    def clear_results(self):
        """Clear all stored results."""
        self.results = {}):
"""Clear all stored results."""
class NetworkManager:
    """Manager for building and controlling node networks."""
    
    def __init__(self):    """Manager for building and controlling node networks."""
        """Initialize a network manager."""
        self.nodes = {}  # Map of node_id -> Node
        self.node_groups = {}  # Map of group_name -> List[node_id]    """Initialize a network manager."""
          # Map of node_id -> Node
    def add_node(self, node: Node) -> 'NetworkManager':up_name -> List[node_id]
        """Add a node to the network."""
        if node.node_id in self.nodes:
            logger.warning(f"NetworkManager replacing existing node {node.node_id}")"""Add a node to the network."""
            
        self.nodes[node.node_id] = nodeger replacing existing node {node.node_id}")
        return self
        
    def remove_node(self, node_id: str) -> Optional[Node]:rn self
        """Remove a node from the network."""
        if node_id in self.nodes:(self, node_id: str) -> Optional[Node]:
            node = self.nodes[node_id]"""Remove a node from the network."""
            
            # Disconnect from all other nodes
            for other_id, other_node in self.nodes.items():
                if node_id in other_node.connections:r nodes
                    other_node.disconnect(node_id, bidirectional=False)for other_id, other_node in self.nodes.items():
                    ections:
            # Remove from all groupsional=False)
            for group_name, node_ids in list(self.node_groups.items()):
                if node_id in node_ids:
                    node_ids.remove(node_id)p_name, node_ids in list(self.node_groups.items()):
                    ds:
            # Remove from the network
            del self.nodes[node_id]
            
            return node.nodes[node_id]
            
        return None
        
    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID."""
        return self.nodes.get(node_id)lf, node_id: str) -> Optional[Node]:
        """Get a node by ID."""
    def connect_nodes(self, node1_id: str, node2_id: str, bidirectional: bool = True) -> bool:
        """Connect two nodes."""
        node1 = self.nodes.get(node1_id)str, node2_id: str, bidirectional: bool = True) -> bool:
        node2 = self.nodes.get(node2_id)"""Connect two nodes."""
        
        if node1 is None or node2 is None:ode2_id)
            logger.warning(f"NetworkManager cannot connect unknown nodes {node1_id}, {node2_id}")
            return Falsee:
                logger.warning(f"NetworkManager cannot connect unknown nodes {node1_id}, {node2_id}")
        node1.connect(node2, bidirectional)
        return True
        de2, bidirectional)
    def disconnect_nodes(self, node1_id: str, node2_id: str, bidirectional: bool = True) -> bool:rn True
        """Disconnect two nodes."""
        node1 = self.nodes.get(node1_id)nodes(self, node1_id: str, node2_id: str, bidirectional: bool = True) -> bool:
        node2 = self.nodes.get(node2_id)"""Disconnect two nodes."""
        
        if node1 is None or node2 is None:2_id)
            logger.warning(f"NetworkManager cannot disconnect unknown nodes {node1_id}, {node2_id}")
            return Falsee:
                logger.warning(f"NetworkManager cannot disconnect unknown nodes {node1_id}, {node2_id}")
        node1.disconnect(node2_id, bidirectional)
        return True
        (node2_id, bidirectional)
    def create_group(self, group_name: str, node_ids: List[str]) -> bool:rn True
        """Create a named group of nodes."""
        # Verify all nodes existp(self, group_name: str, node_ids: List[str]) -> bool:
        for node_id in node_ids:"""Create a named group of nodes."""
            if node_id not in self.nodes:
                logger.warning(f"NetworkManager cannot create group with unknown node {node_id}")
                return Falself.nodes:
                "NetworkManager cannot create group with unknown node {node_id}")
        self.node_groups[group_name] = node_ids.copy()
        return True
        up_name] = node_ids.copy()
    def add_to_group(self, group_name: str, node_id: str) -> bool:rue
        """Add a node to a group."""
        if group_name not in self.node_groups:p(self, group_name: str, node_id: str) -> bool:
            logger.warning(f"NetworkManager cannot add to unknown group {group_name}")"""Add a node to a group."""
            return False
            Manager cannot add to unknown group {group_name}")
        if node_id not in self.nodes:
            logger.warning(f"NetworkManager cannot add unknown node {node_id} to group")
            return Falsen self.nodes:
            logger.warning(f"NetworkManager cannot add unknown node {node_id} to group")
        if node_id not in self.node_groups[group_name]:
            self.node_groups[group_name].append(node_id)
            n self.node_groups[group_name]:
        return Trueself.node_groups[group_name].append(node_id)
        
    def remove_from_group(self, group_name: str, node_id: str) -> bool:
        """Remove a node from a group."""
        if group_name not in self.node_groups:_group(self, group_name: str, node_id: str) -> bool:
            logger.warning(f"NetworkManager cannot remove from unknown group {group_name}")"""Remove a node from a group."""
            return False
            logger.warning(f"NetworkManager cannot remove from unknown group {group_name}")
            return Falseame]:
            self.node_groups[group_name].remove(node_id)
            lf.node_groups[group_name]:
        return Trueself.node_groups[group_name].remove(node_id)
        
    def get_group(self, group_name: str) -> List[str]:
        """Get the node IDs in a group."""
        return self.node_groups.get(group_name, []).copy()elf, group_name: str) -> List[str]:
        """Get the node IDs in a group."""
    def broadcast_to_group(self, group_name: str, message: Message) -> int:py()
        """Broadcast a message to all nodes in a group."""
        if group_name not in self.node_groups: Message) -> int:
            logger.warning(f"NetworkManager cannot broadcast to unknown group {group_name}")"""Broadcast a message to all nodes in a group."""
            return 0
            st to unknown group {group_name}")
        success_count = 0
        
        for node_id in self.node_groups[group_name]:t = 0
            node = self.nodes.get(node_id)
            if node and node.receive(message):lf.node_groups[group_name]:
                success_count += 1    node = self.nodes.get(node_id)
                
        return success_count
        
    def process_all(self) -> Dict[str, int]:
        """Process all messages in all nodes."""
        results = {} Dict[str, int]:
        """Process all messages in all nodes."""
        for node_id, node in self.nodes.items():
            count = node.process_all()
            results[node_id] = count node in self.nodes.items():
                count = node.process_all()
        return results
        
    def activate_all(self):
        """Activate all nodes."""
        for node in self.nodes.values():elf):
            node.