<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kaleidoscope AI - Quantum Visualization</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        :root {
            --primary: #3a0ca3;
            --secondary: #4cc9f0;
            --accent: #f72585;
            --dark: #101020;
            --light: #ffffff;
            --success: #06d6a0;
            --warning: #ffd166;
            --danger: #ef476f;
        }
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Arial', sans-serif;
        }
        body {
            overflow: hidden;
            background: radial-gradient(circle at center, #1a1a3a 0%, #000020 100%);
        }
        #visualization {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 1;
        }
        #particle-layer {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 2;
            pointer-events: none;
            opacity: 0.7;
        }
        #ui-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 3;
            pointer-events: none;
        }
        .ui-element {
            pointer-events: auto;
        }
        #control-panel {
            position: fixed;
            top: 20px;
            right: 20px;
            width: 300px;
            background: rgba(16, 16, 32, 0.8);
            backdrop-filter: blur(10px);
            border-radius: 10px;
            border: 1px solid var(--secondary);
            padding: 15px;
            color: var(--light);
            z-index: 10;
            box-shadow: 0 0 20px rgba(76, 201, 240, 0.3);
            transform: translateX(320px);
            transition: transform 0.3s ease;
        }
        #control-panel.visible {
            transform: translateX(0);
        }
        #control-panel h3 {
            margin-bottom: 15px;
            color: var(--secondary);
            font-size: 16px;
            border-bottom: 1px solid rgba(76, 201, 240, 0.3);
            padding-bottom: 8px;
        }
        .control-row {
            margin-bottom: 12px;
        }
        .control-row label {
            display: block;
            margin-bottom: 5px;
            font-size: 14px;
            color: rgba(255, 255, 255, 0.8);
        }
        .slider-container {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .slider-container input {
            flex: 1;
        }
        .slider-container .value {
            width: 40px;
            text-align: center;
            font-size: 12px;
            color: var(--secondary);
        }
        input[type="range"] {
            -webkit-appearance: none;
            width: 100%;
            height: 5px;
            background: rgba(76, 201, 240, 0.2);
            border-radius: 3px;
        }
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            height: 15px;
            width: 15px;
            border-radius: 50%;
            background: var(--secondary);
            cursor: pointer;
        }
        .button-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            margin-top: 15px;
        }
        .ctrl-btn {
            background: rgba(58, 12, 163, 0.4);
            border: 1px solid var(--primary);
            color: var(--light);
            padding: 8px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
            transition: all 0.2s;
        }
        .ctrl-btn:hover {
            background: var(--primary);
        }
        .ctrl-btn.active {
            background: var(--primary);
            border-color: var(--secondary);
        }
        .colorbox {
            width: 20px;
            height: 20px;
            display: inline-block;
            border-radius: 3px;
            cursor: pointer;
            border: 2px solid rgba(255, 255, 255, 0.2);
            transition: transform 0.2s;
        }
        .colorbox:hover {
            transform: scale(1.1);
            border-color: white;
        }
        .colorbox.active {
            border-color: white;
        }
        .colors-row {
            display: flex;
            gap: 8px;
            margin-top: 8px;
        }
        #panel-toggle {
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(16, 16, 32, 0.8);
            border-radius: 50%;
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            z-index: 11;
            border: 1px solid var(--secondary);
            color: var(--secondary);
            box-shadow: 0 0 10px rgba(76, 201, 240, 0.3);
        }
        #stats-panel {
            position: fixed;
            bottom: 20px;
            left: 20px;
            background: rgba(16, 16, 32, 0.8);
            backdrop-filter: blur(10px);
            border-radius: 10px;
            border: 1px solid var(--secondary);
            padding: 10px 15px;
            color: var(--light);
            font-family: monospace;
            font-size: 12px;
            z-index: 10;
        }
        .stat-row {
            display: flex;
            justify-content: space-between;
            gap: 15px;
            margin-bottom: 4px;
        }
        .stat-value {
            color: var(--secondary);
        }
        #toast {
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%) translateY(100px);
            background: rgba(16, 16, 32, 0.9);
            border-left: 3px solid var(--secondary);
            color: var(--light);
            padding: 12px 20px;
            border-radius: 5px;
            font-size: 14px;
            transition: transform 0.3s ease;
            z-index: 100;
        }
        #toast.visible {
            transform: translateX(-50%) translateY(0);
        }
        #loading {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: radial-gradient(circle at center, #1a1a3a 0%, #000020 100%);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            z-index: 1000;
            transition: opacity 0.5s ease;
        }
        #loading h2 {
            color: var(--light);
            margin-bottom: 20px;
            font-size: 24px;
        }
        .loading-cube-container {
            width: 100px;
            height: 100px;
            perspective: 800px;
            margin-bottom: 30px;
        }
        .loading-cube {
            width: 100%;
            height: 100%;
            position: relative;
            transform-style: preserve-3d;
            transform: translateZ(-50px);
            animation: loading-rotate 3s infinite linear;
        }
        .loading-face {
            position: absolute;
            width: 100px;
            height: 100px;
            border: 2px solid var(--secondary);
            background: rgba(76, 201, 240, 0.1);
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--secondary);
            font-size: 24px;
        }
        .loading-face:nth-child(1) { transform: rotateY(0deg) translateZ(50px); }
        .loading-face:nth-child(2) { transform: rotateY(90deg) translateZ(50px); }
        .loading-face:nth-child(3) { transform: rotateY(180deg) translateZ(50px); }
        .loading-face:nth-child(4) { transform: rotateY(-90deg) translateZ(50px); }
        .loading-face:nth-child(5) { transform: rotateX(90deg) translateZ(50px); }
        .loading-face:nth-child(6) { transform: rotateX(-90deg) translateZ(50px); }
        @keyframes loading-rotate {
            0% { transform: translateZ(-50px) rotateX(0deg) rotateY(0deg); }
            100% { transform: translateZ(-50px) rotateX(360deg) rotateY(360deg); }
        }
        #progress-bar {
            width: 300px;
            height: 4px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 2px;
            overflow: hidden;
            margin-top: 20px;
        }
        #progress-fill {
            height: 100%;
            width: 0;
            background: linear-gradient(90deg, var(--accent), var(--secondary));
            transition: width 0.5s ease;
        }
        #loading-text {
            color: rgba(255, 255, 255, 0.7);
            font-size: 14px;
            margin-top: 10px;
            font-family: monospace;
        }
        #context-menu {
            position: absolute;
            background: rgba(16, 16, 32, 0.9);
            border: 1px solid var(--secondary);
            border-radius: 5px;
            padding: 5px 0;
            min-width: 150px;
            z-index: 100;
            display: none;
        }
        .context-item {
            padding: 8px 12px;
            cursor: pointer;
            font-size: 14px;
            color: var(--light);
        }
        .context-item:hover {
            background: rgba(76, 201, 240, 0.2);
        }
        .context-divider {
            height: 1px;
            background: rgba(255, 255, 255, 0.1);
            margin: 5px 0;
        }
        #modal {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1000;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.3s ease;
        }
        #modal.visible {
            opacity: 1;
            pointer-events: auto;
        }
        .modal-content {
            background: rgba(26, 26, 46, 0.95);
            border: 1px solid var(--secondary);
            border-radius: 10px;
            padding: 20px;
            width: 90%;
            max-width: 500px;
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.5);
        }
        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            border-bottom: 1px solid rgba(76, 201, 240, 0.3);
            padding-bottom: 10px;
        }
        .modal-title {
            color: var(--secondary);
            font-size: 18px;
        }
        .modal-close {
            color: var(--light);
            background: none;
            border: none;
            font-size: 20px;
            cursor: pointer;
        }
        .modal-body {
            color: var(--light);
            margin-bottom: 20px;
            max-height: 60vh;
            overflow-y: auto;
        }
        .modal-footer {
            display: flex;
            justify-content: flex-end;
            gap: 10px;
        }
        .modal-btn {
            background: rgba(58, 12, 163, 0.4);
            border: 1px solid var(--primary);
            color: var(--light);
            padding: 8px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
        }
        .modal-btn:hover {
            background: var(--primary);
        }
        .modal-btn.primary {
            background: var(--primary);
            border-color: var(--secondary);
        }
        .modal-btn.primary:hover {
            background: var(--secondary);
            color: var(--dark);
        }
        @media (max-width: 768px) {
            #control-panel {
                width: 250px;
            }
            .button-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div id="loading">
        <div class="loading-cube-container">
            <div class="loading-cube">
                <div class="loading-face"><i class="fas fa-atom"></i></div>
                <div class="loading-face"><i class="fas fa-brain"></i></div>
                <div class="loading-face"><i class="fas fa-project-diagram"></i></div>
                <div class="loading-face"><i class="fas fa-cube"></i></div>
                <div class="loading-face"><i class="fas fa-network-wired"></i></div>
                <div class="loading-face"><i class="fas fa-microchip"></i></div>
            </div>
        </div>
        <h2>Initializing Quantum Visualization</h2>
        <div id="progress-bar">
            <div id="progress-fill"></div>
        </div>
        <div id="loading-text">Loading core components...</div>
    </div>
    
    <canvas id="visualization"></canvas>
    <canvas id="particle-layer"></canvas>
    
    <div id="ui-overlay">
        <div id="panel-toggle" class="ui-element">
            <i class="fas fa-cog"></i>
        </div>
        
        <div id="control-panel" class="ui-element">
            <h3>Quantum Cube Controls</h3>
            
            <div class="control-row">
                <label>Cube Size</label>
                <div class="slider-container">
                    <input type="range" id="size-slider" min="5" max="30" value="15">
                    <div class="value" id="size-value">15</div>
                </div>
            </div>
            
            <div class="control-row">
                <label>Node Density</label>
                <div class="slider-container">
                    <input type="range" id="density-slider" min="1" max="20" value="8">
                    <div class="value" id="density-value">8</div>
                </div>
            </div>
            
            <div class="control-row">
                <label>Energy Level</label>
                <div class="slider-container">
                    <input type="range" id="energy-slider" min="0" max="100" value="60">
                    <div class="value" id="energy-value">60</div>
                </div>
            </div>
            
            <div class="control-row">
                <label>Connection Threshold</label>
                <div class="slider-container">
                    <input type="range" id="connection-slider" min="1" max="15" value="5">
                    <div class="value" id="connection-value">5</div>
                </div>
            </div>
            
            <div class="control-row">
                <label>Rotation Speed</label>
                <div class="slider-container">
                    <input type="range" id="rotation-slider" min="0" max="10" value="2">
                    <div class="value" id="rotation-value">2</div>
                </div>
            </div>
            
            <div class="control-row">
                <label>Node Color</label>
                <div class="colors-row">
                    <div class="colorbox active" data-color="#f72585" style="background-color: #f72585;"></div>
                    <div class="colorbox" data-color="#4cc9f0" style="background-color: #4cc9f0;"></div>
                    <div class="colorbox" data-color="#7209b7" style="background-color: #7209b7;"></div>
                    <div class="colorbox" data-color="#06d6a0" style="background-color: #06d6a0;"></div>
                    <div class="colorbox" data-color="#ffd166" style="background-color: #ffd166;"></div>
                </div>
            </div>
            
            <div class="button-grid">
                <button class="ctrl-btn" id="reset-btn">Reset</button>
                <button class="ctrl-btn" id="add-nodes-btn">Add Nodes</button>
                <button class="ctrl-btn" id="entangle-btn">Entangle</button>
                <button class="ctrl-btn" id="explosion-btn">Quantum Burst</button>
                <button class="ctrl-btn" id="wireframe-btn">Toggle Wireframe</button>
                <button class="ctrl-btn" id="glow-btn">Toggle Glow</button>
                <button class="ctrl-btn" id="save-btn">Save Image</button>
                <button class="ctrl-btn" id="fullscreen-btn">Fullscreen</button>
            </div>
        </div>
        
        <div id="stats-panel" class="ui-element">
            <div class="stat-row">
                <div class="stat-label">FPS:</div>
                <div class="stat-value" id="fps-value">0</div>
            </div>
            <div class="stat-row">
                <div class="stat-label">Nodes:</div>
                <div class="stat-value" id="nodes-value">0</div>
            </div>
            <div class="stat-row">
                <div class="stat-label">Connections:</div>
                <div class="stat-value" id="connections-value">0</div>
            </div>
            <div class="stat-row">
                <div class="stat-label">Energy:</div>
                <div class="stat-value" id="energy-stat-value">60%</div>
            </div>
        </div>
    </div>
    
    <div id="toast"></div>
    
    <div id="context-menu" class="ui-element">
        <div class="context-item" id="ctx-add-node">Add Node Here</div>
        <div class="context-item" id="ctx-clear-area">Clear Nearby Nodes</div>
        <div class="context-divider"></div>
        <div class="context-item" id="ctx-explode-from-here">Burst From Here</div>
        <div class="context-item" id="ctx-create-cluster">Create Cluster</div>
    </div>
    
    <div id="modal">
        <div class="modal-content">
            <div class="modal-header">
                <div class="modal-title">Quantum Visualization</div>
                <button class="modal-close">&times;</button>
            </div>
            <div class="modal-body" id="modal-body">
                Modal content here
            </div>
            <div class="modal-footer">
                <button class="modal-btn" id="modal-cancel">Cancel</button>
                <button class="modal-btn primary" id="modal-confirm">Confirm</button>
            </div>
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
    (() => {
        // Core system state
        const state = {
            cube: {
                size: 15,
                density: 8,
                rotationSpeed: 2,
                wireframe: true,
                color: 0x4cc9f0
            },
            nodes: {
                color: 0xf72585,
                glowEnabled: true,
                connectionThreshold: 5,
                entangled: false
            },
            energy: 60,
            running: true,
            mousePosition: { x: 0, y: 0 },
            mouseWorldPosition: new THREE.Vector3(),
            fpsSamples: [],
            showContextMenu: false,
            contextMenuPosition: { x: 0, y: 0 },
            selectedNode: null,
            explosion: {
                active: false,
                position: new THREE.Vector3(),
                timer: 0,
                duration: 3000, // ms
                strength: 1
            },
            lastClickTime: 0,
            isDragging: false,
            previousMousePosition: { x: 0, y: 0 }
        };
        
        // Collection of 3D objects
        const objects = {
            nodes: [],
            connections: [],
            cube: null,
            cubeWireframe: null,
            raycaster: new THREE.Raycaster(),
            scene: null,
            camera: null,
            renderer: null,
            particleCanvas: null,
            particleContext: null,
            particles: [],
            clock: new THREE.Clock(),
            pointLight: null
        };
        
        // UI interactions and state
        const ui = {
            controlPanelVisible: false,
            sliders: {},
            values: {},
            buttons: {},
            toastTimeout: null,
            modalCallback: null,
            colorBoxes: [],
            time: 0
        };
        
        /**
         * Initialize the 3D visualization system
         */
        function initVisualization() {
            try {
                // Setup main ThreeJS scene
                objects.scene = new THREE.Scene();
                objects.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
                objects.camera.position.z = 30;
                
                // Create renderer with antialiasing and transparency
                objects.renderer = new THREE.WebGLRenderer({
                    canvas: document.getElementById('visualization'),
                    antialias: true,
                    alpha: true
                });
                objects.renderer.setSize(window.innerWidth, window.innerHeight);
                objects.renderer.setClearColor(0x000000, 0);
                objects.renderer.setPixelRatio(window.devicePixelRatio);
                
                // Setup particle canvas
                objects.particleCanvas = document.getElementById('particle-layer');
                objects.particleContext = objects.particleCanvas.getContext('2d');
                objects.particleCanvas.width = window.innerWidth;
                objects.particleCanvas.height = window.innerHeight;
                
                // Setup lighting
                setupLighting();
                
                // Create initial cube
                createQuantumCube();
                
                // Setup event handlers
                setupEventHandlers();
                
                // Begin the animation loop
                animate();
                
                showToast("Quantum visualization initialized");
                
                // Show control panel
                document.getElementById('control-panel').classList.add('visible');
                ui.controlPanelVisible = true;
            } catch (error) {
                console.error("Error initializing visualization:", error);
                showToast("Error initializing visualization", "error");
            }
        }
        
        /**
         * Setup lighting for the scene
         */
        function setupLighting() {
            // Ambient light
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.3);
            objects.scene.add(ambientLight);
            
            // Directional light
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.7);
            directionalLight.position.set(5, 5, 5);
            objects.scene.add(directionalLight);
            
            // Point light at center
            const pointLight = new THREE.PointLight(0x4cc9f0, 1, 100);
            pointLight.position.set(0, 0, 0);
            objects.scene.add(pointLight);
            
            // Store the point light for energy adjustments
            objects.pointLight = pointLight;
        }
        
        /**
         * Create the quantum cube and nodes
         */
        function createQuantumCube() {
            try {
                // Remove existing cube and nodes
                if (objects.cube) objects.scene.remove(objects.cube);
                if (objects.cubeWireframe) objects.scene.remove(objects.cubeWireframe);
                clearNodes();
                
                // Create cube geometry
                const geometry = new THREE.BoxGeometry(
                    state.cube.size,
                    state.cube.size,
                    state.cube.size
                );
                
                // Create wireframe
                const wireframeGeometry = new THREE.EdgesGeometry(geometry);
                const wireframeMaterial = new THREE.LineBasicMaterial({
                    color: state.cube.color,
                    transparent: true,
                    opacity: 0.7
                });
                
                objects.cubeWireframe = new THREE.LineSegments(wireframeGeometry, wireframeMaterial);
                objects.cubeWireframe.visible = state.cube.wireframe;
                objects.scene.add(objects.cubeWireframe);
                
                // Create nodes
                createNodes();
                
                // Update UI stats
                updateStats();
            } catch (error) {
                console.error("Error creating quantum cube:", error);
                showToast("Error creating quantum structure", "error");
            }
        }
        
        /**
         * Create quantum nodes inside the cube
         */
        function createNodes() {
            try {
                const nodeCount = Math.pow(state.cube.density, 2);
                const halfSize = state.cube.size / 2 * 0.8; // 80% of half-size to keep inside
                
                for (let i = 0; i < nodeCount; i++) {
                    // Random position within cube bounds
                    const position = new THREE.Vector3(
                        (Math.random() * 2 - 1) * halfSize,
                        (Math.random() * 2 - 1) * halfSize,
                        (Math.random() * 2 - 1) * halfSize
                    );
                    
                    addNode(position);
                }
                
                showToast(`Created ${nodeCount} quantum nodes`);
            } catch (error) {
                console.error("Error creating nodes:", error);
                showToast("Error creating quantum nodes", "error");
            }
        }
        
        /**
         * Add a single node at the specified position
         */
        function addNode(position) {
            try {
                // Create sphere geometry for the node
                const geometry = new THREE.SphereGeometry(0.4, 16, 16);
                
                // Create material with physical properties
                const material = new THREE.MeshPhysicalMaterial({
                    color: state.nodes.color,
                    metalness: 0.8,
                    roughness: 0.2,
                    emissive: state.nodes.color,
                    emissiveIntensity: state.nodes.glowEnabled ? 0.5 : 0,
                    transparent: true,
                    opacity: 0.9
                });
                
                // Create the node mesh
                const node = new THREE.Mesh(geometry, material);
                node.position.copy(position);
                
                // Add physics properties to the node
                node.userData = {
                    velocity: new THREE.Vector3(
                        (Math.random() - 0.5) * 0.05,
                        (Math.random() - 0.5) * 0.05,
                        (Math.random() - 0.5) * 0.05
                    ),
                    energy: Math.random() * 0.5 + 0.5,
                    connections: [],
                    entangled: false,
                    originalColor: state.nodes.color,
                    phase: Math.random() * Math.PI * 2
                };
                
                // Add to scene and track
                objects.scene.add(node);
                objects.nodes.push(node);
                
                return node;
            } catch (error) {
                console.error("Error adding node:", error);
                return null;
            }
        }
        
        /**
         * Create connections between nodes based on distance
         */
        function entangleNodes() {
            try {
                // Remove existing connections
                clearConnections();
                
                // Reset node connection properties
                objects.nodes.forEach(node => {
                    node.userData.connections = [];
                    node.userData.entangled = false;
                });
                
                // Create connections based on distance
                for (let i = 0; i < objects.nodes.length; i++) {
                    for (let j = i + 1; j < objects.nodes.length; j++) {
                        const node1 = objects.nodes[i];
                        const node2 = objects.nodes[j];
                        
                        const distance = node1.position.distanceTo(node2.position);
                        
                        if (distance < state.nodes.connectionThreshold) {
                            createConnection(node1, node2);
                        }
                    }
                }
                
                state.nodes.entangled = objects.connections.length > 0;
                showToast(`Created ${objects.connections.length} quantum connections`);
                updateStats();
            } catch (error) {
                console.error("Error entangling nodes:", error);
                showToast("Error creating quantum entanglement", "error");
            }
        }
        
        /**
         * Clear all connections
         */
        function clearConnections() {
            objects.connections.forEach(conn => {
                objects.scene.remove(conn);
            });
            objects.connections = [];
        }
        
        /**
         * Create a connection between two nodes
         */
        function createConnection(node1, node2) {
            try {
                // Create line between nodes
                const points = [
                    node1.position.clone(),
                    node2.position.clone()
                ];
                
                const geometry = new THREE.BufferGeometry().setFromPoints(points);
                
                const material = new THREE.LineBasicMaterial({
                    color: state.cube.color,
                    transparent: true,
                    opacity: 0.5
                });
                
                const connection = new THREE.Line(geometry, material);
                
                // Store connection references
                connection.userData = {
                    node1: node1,
                    node2: node2
                };
                
                node1.userData.connections.push(connection);
                node2.userData.connections.push(connection);
                node1.userData.entangled = true;
                node2.userData.entangled = true;
                
                objects.scene.add(connection);
                objects.connections.push(connection);
                
                return connection;
            } catch (error) {
                console.error("Error creating connection:", error);
                return null;
            }
        }
        
        /**
         * Clear all nodes from the scene
         */
        function clearNodes() {
            try {
                // Remove connections
                clearConnections();
                
                // Remove nodes
                objects.nodes.forEach(node => {
                    objects.scene.remove(node);
                });
                objects.nodes = [];
                
                updateStats();
            } catch (error) {
                console.error("Error clearing nodes:", error);
            }
        }
        
        /**
         * Create an explosion effect at a specific position
         */
        function createExplosionEffect(position, strength = 1) {
            try {
                // Set explosion state
                state.explosion.active = true;
                state.explosion.position.copy(position);
                state.explosion.timer = 0;
                state.explosion.strength = strength;
                
                // Create particle burst
                const particleCount = 50 * strength;
                
                for (let i = 0; i < particleCount; i++) {
                    const size = Math.random() * 0.3 + 0.1;
                    
                    // Random direction from center point
                    const direction = new THREE.Vector3(
                        Math.random() * 2 - 1,
                        Math.random() * 2 - 1,
                        Math.random() * 2 - 1
                    ).normalize();
                    
                    // Random speed
                    const speed = (Math.random() * 0.5 + 0.5) * strength;
                    
                    // Particle position starts at explosion center
                    const particle = {
                        position: position.clone(),
                        velocity: direction.clone().multiplyScalar(speed),
                        size: size,
                        color: state.nodes.color,
                        opacity: 1,
                        life: 1, // 0-1 lifecycle
                        decay: 0.01 + Math.random() * 0.02 // How fast it decays per frame
                    };
                    
                    objects.particles.push(particle);
                }
                
                showToast("Quantum energy burst initiated");
            } catch (error) {
                console.error("Error creating explosion:", error);
            }
        }
        
        /**
         * Draw 2D particles on the particle canvas
         */
        function drawParticles() {
            try {
                const ctx = objects.particleContext;
                const canvas = objects.particleCanvas;
                
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                
                for (let i = objects.particles.length - 1; i >= 0; i--) {
                    const particle = objects.particles[i];
                    
                    // Update particle position
                    particle.position.add(particle.velocity);
                    
                    // Update lifecycle
                    particle.life -= particle.decay;
                    
                    // Remove dead particles
                    if (particle.life <= 0) {
                        objects.particles.splice(i, 1);
                        continue;
                    }
                    
                    // Project 3D position to 2D screen space
                    const vector = particle.position.clone().project(objects.camera);
                    
                    const x = (vector.x * 0.5 + 0.5) * canvas.width;
                    const y = (-vector.y * 0.5 + 0.5) * canvas.height;
                    
                    // Draw particle only if it's on-screen
                    if (x >= 0 && x <= canvas.width && y >= 0 && y <= canvas.height) {
                        // Get color components
                        const r = (particle.color >> 16) & 255;
                        const g = (particle.color >> 8) & 255;
                        const b = particle.color & 255;
                        
                        const radius = particle.size * 10 * particle.life;
                        
                        // Draw particle
                        ctx.beginPath();
                        ctx.arc(x, y, radius, 0, Math.PI * 2);
                        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${particle.life})`;
                        ctx.fill();
                    }
                }
            } catch (error) {
                console.error("Error drawing particles:", error);
            }
        }
        
        /**
         * Update all node positions and connections
         */
        function updateNodes(deltaTime) {
            try {
                const halfSize = state.cube.size / 2;
                
                objects.nodes.forEach(node => {
                    // If explosion is active, apply force from explosion center
                    if (state.explosion.active) {
                        const direction = node.position.clone().sub(state.explosion.position);
                        const distance = direction.length();
                        
                        // If within explosion radius, apply force
                        if (distance < state.cube.size) {
                            // Normalize direction and apply force inversely proportional to distance
                            direction.normalize();
                            const force = (1 - distance / state.cube.size) * state.explosion.strength;
                            
                            // Add to velocity
                            node.userData.velocity.add(
                                direction.multiplyScalar(force * 0.1)
                            );
                        }
                    }
                    
                    // Apply velocity to position
                    node.position.add(node.userData.velocity);
                    
                    // Bounce off cube walls
                    if (Math.abs(node.position.x) > halfSize) {
                        node.userData.velocity.x *= -0.9; // Damping factor
                        node.position.x = Math.sign(node.position.x) * halfSize;
                    }
                    
                    if (Math.abs(node.position.y) > halfSize) {
                        node.userData.velocity.y *= -0.9;
                        node.position.y = Math.sign(node.position.y) * halfSize;
                    }
                    
                    if (Math.abs(node.position.z) > halfSize) {
                        node.userData.velocity.z *= -0.9;
                        node.position.z = Math.sign(node.position.z) * halfSize;
                    }
                    
                    // Apply energy to node pulsation
                    const energyFactor = state.energy / 100;
                    const time = objects.clock.elapsedTime;
                    const pulseFactor = 1 + Math.sin(time * 2 + node.userData.phase) * 0.2 * energyFactor * node.userData.energy;
                    
                    node.scale.set(pulseFactor, pulseFactor, pulseFactor);
                    
                    // Apply energy to glow
                    if (state.nodes.glowEnabled) {
                        node.material.emissiveIntensity = 0.3 + energyFactor * 0.7 * node.userData.energy;
                    }
                    
                    // Slowly stabilize velocity (damping)
                    node.userData.velocity.multiplyScalar(0.99);
                });
                
                // Update connections to follow nodes
                objects.connections.forEach(connection => {
                    const { node1, node2 } = connection.userData;
                    
                    // Make sure both connected nodes still exist in scene
                    if (!objects.nodes.includes(node1) || !objects.nodes.includes(node2)) {
                        objects.scene.remove(connection);
                        
                        // Remove connection from the other node's connection list
                        if (objects.nodes.includes(node1)) {
                            const idx = node1.userData.connections.indexOf(connection);
                            if (idx > -1) node1.userData.connections.splice(idx, 1);
                        }
                        
                        if (objects.nodes.includes(node2)) {
                            const idx = node2.userData.connections.indexOf(connection);
                            if (idx > -1) node2.userData.connections.splice(idx, 1);
                        }
                        
                        return;
                    }
                    
                    // Update line geometry to match new node positions
                    const positions = connection.geometry.attributes.position.array;
                    
                    positions[0] = node1.position.x;
                    positions[1] = node1.position.y;
                    positions[2] = node1.position.z;
                    
                    positions[3] = node2.position.x;
                    positions[4] = node2.position.y;
                    positions[5] = node2.position.z;
                    
                    connection.geometry.attributes.position.needsUpdate = true;
                    
                    // Adjust connection opacity based on distance
                    const distance = node1.position.distanceTo(node2.position);
                    
                    // Break connections that stretch too far
                    if (distance > state.nodes.connectionThreshold * 2) {
                        connection.material.opacity = 0.1;
                    } else {
                        connection.material.opacity = 0.5 * (1 - distance / (state.nodes.connectionThreshold * 2));
                    }
                });
                
                // Clean up broken connections
                for (let i = objects.connections.length - 1; i >= 0; i--) {
                    const connection = objects.connections[i];
                    
                    if (!objects.scene.children.includes(connection)) {
                        objects.connections.splice(i, 1);
                    }
                }
            } catch (error) {
                console.error("Error updating nodes:", error);
            }
        }
        
        /**
         * Main animation loop
         */
        function animate() {
            if (!state.running) return;
            
            requestAnimationFrame(animate);
            
            try {
                // Calculate delta time
                const deltaTime = objects.clock.getDelta();
                
                // Update FPS counter
                updateFps();
                
                // Apply rotation to cube
                if (objects.cubeWireframe && state.cube.rotationSpeed > 0 && !state.isDragging) {
                    objects.cubeWireframe.rotation.y += state.cube.rotationSpeed * 0.001;
                    objects.cubeWireframe.rotation.x += state.cube.rotationSpeed * 0.0005;
                    
                    // Rotate all nodes with the cube if not dragging
                    const rotationMatrix = new THREE.Matrix4().makeRotationFromEuler(
                        new THREE.Euler(
                            state.cube.rotationSpeed * 0.0005,
                            state.cube.rotationSpeed * 0.001,
                            0,
                            'XYZ'
                        )
                    );
                    
                    objects.nodes.forEach(node => {
                        node.position.applyMatrix4(rotationMatrix);
                    });
                }
                
                // Update explosion state
                if (state.explosion.active) {
                    state.explosion.timer += deltaTime * 1000;
                    
                    if (state.explosion.timer >= state.explosion.duration) {
                        state.explosion.active = false;
                    }
                }
                
                // Update nodes
                updateNodes(deltaTime);
                
                // Draw particles
                drawParticles();
                
                // Render scenes
                objects.renderer.render(objects.scene, objects.camera);
                
                // Update global timer
                ui.time = objects.clock.elapsedTime;
            } catch (error) {
                console.error("Error in animation loop:", error);
                state.running = false;
                showToast("Rendering error occurred", "error");
            }
        }
        
        /**
         * Update the UI stats display
         */
        function updateStats() {
            document.getElementById('nodes-value').textContent = objects.nodes.length;
            document.getElementById('connections-value').textContent = objects.connections.length;
            document.getElementById('energy-stat-value').textContent = `${state.energy}%`;
        }
        
        /**
         * Calculate and update FPS display
         */
        function updateFps() {
            const now = performance.now();
            
            while (state.fpsSamples.length > 0 && state.fpsSamples[0] <= now - 1000) {
                state.fpsSamples.shift();
            }
            
            state.fpsSamples.push(now);
            document.getElementById('fps-value').textContent = state.fpsSamples.length;
        }
        
        /**
         * Display a toast notification
         */
        function showToast(message, type = 'info') {
            const toast = document.getElementById('toast');
            toast.textContent = message;
            
            // Set color based on type
            if (type === 'error') {
                toast.style.borderLeftColor = '#ef476f';
            } else if (type === 'warning') {
                toast.style.borderLeftColor = '#ffd166';
            } else {
                toast.style.borderLeftColor = '#4cc9f0';
            }
            
            toast.classList.add('visible');
            
            // Clear any existing timeout
            if (ui.toastTimeout) {
                clearTimeout(ui.toastTimeout);
            }
            
            // Hide after 3 seconds
            ui.toastTimeout = setTimeout(() => {
                toast.classList.remove('visible');
            }, 3000);
        }
        
        /**
         * Convert hex color string to integer
         */
        function hexToInt(hex) {
            return parseInt(hex.replace('#', '0x'));
        }
        
        /**
         * Setup event handlers for interaction
         */
        function setupEventHandlers() {
            // Window resize
            window.addEventListener('resize', () => {
                // Update renderer sizes
                objects.renderer.setSize(window.innerWidth, window.innerHeight);
                
                // Update particle canvas
                objects.particleCanvas.width = window.innerWidth;
                objects.particleCanvas.height = window.innerHeight;
                
                // Update camera aspect ratio
                objects.camera.aspect = window.innerWidth / window.innerHeight;
                objects.camera.updateProjectionMatrix();
            });
            
            // Track mouse position for raycasting
            document.addEventListener('mousemove', (event) => {
                state.mousePosition.x = (event.clientX / window.innerWidth) * 2 - 1;
                state.mousePosition.y = -(event.clientY / window.innerHeight) * 2 + 1;
                
                // Project mouse position to 3D space
                objects.raycaster.setFromCamera(state.mousePosition, objects.camera);
                const intersects = objects.raycaster.intersectObject(objects.cubeWireframe);
                
                if (intersects.length > 0) {
                    state.mouseWorldPosition.copy(intersects[0].point);
                }
                
                // Handle dragging rotation
                if (state.isDragging) {
                    const deltaMove = {
                        x: event.clientX - state.previousMousePosition.x,
                        y: event.clientY - state.previousMousePosition.y
                    };
                    
                    if (objects.cubeWireframe) {
                        objects.cubeWireframe.rotation.y += deltaMove.x * 0.01;
                        objects.cubeWireframe.rotation.x += deltaMove.y * 0.01;
                        
                        // Rotate all nodes together with the cube
                        const rotationMatrix = new THREE.Matrix4().makeRotationFromEuler(
                            new THREE.Euler(deltaMove.y * 0.01, deltaMove.x * 0.01, 0, 'XYZ')
                        );
                        
                        objects.nodes.forEach(node => {
                            node.position.applyMatrix4(rotationMatrix);
                        });
                    }
                }
                
                state.previousMousePosition = {
                    x: event.clientX,
                    y: event.clientY
                };
            });
            
            // Mouse click to add nodes
            document.addEventListener('mousedown', (event) => {
                // Skip if right click (context menu)
                if (event.button === 2) return;
                
                state.isDragging = true;
                state.previousMousePosition = {
                    x: event.clientX,
                    y: event.clientY
                };
                
                // Handle double click
                const now = Date.now();
                const timeBetweenClicks = now - state.lastClickTime;
                state.lastClickTime = now;
                
                if (timeBetweenClicks < 300) { // Double click threshold
                    // Project mouse position to 3D space
                    objects.raycaster.setFromCamera(state.mousePosition, objects.camera);
                    const intersects = objects.raycaster.intersectObject(objects.cubeWireframe);
                    
                    if (intersects.length > 0) {
                        const node = addNode(intersects[0].point);
                        if (node) {
                            // Create small particle effect
                            createExplosionEffect(intersects[0].point, 0.3);
                            updateStats();
                        }
                    }
                }
            });
            
            document.addEventListener('mouseup', () => {
                state.isDragging = false;
            });
            
            // Mouse wheel to zoom
            document.addEventListener('wheel', (event) => {
                const zoomSpeed = 0.1;
                objects.camera.position.z += event.deltaY * 0.01;
                
                // Clamp zoom distance
                objects.camera.position.z = Math.max(5, Math.min(50, objects.camera.position.z));
            });
            
            // Context menu
            document.addEventListener('contextmenu', (event) => {
                event.preventDefault();
                
                // Show context menu at click position
                const contextMenu = document.getElementById('context-menu');
                contextMenu.style.left = `${event.clientX}px`;
                contextMenu.style.top = `${event.clientY}px`;
                contextMenu.style.display = 'block';
                
                state.contextMenuPosition = { x: event.clientX, y: event.clientY };
                
                // Check if we're clicking on a node
                objects.raycaster.setFromCamera(state.mousePosition, objects.camera);
                const intersects = objects.raycaster.intersectObjects(objects.nodes);
                
                if (intersects.length > 0) {
                    state.selectedNode = intersects[0].object;
                } else {
                    state.selectedNode = null;
                }
            });
            
            // Hide context menu when clicking elsewhere
            document.addEventListener('click', (event) => {
                const contextMenu = document.getElementById('context-menu');
                if (contextMenu.style.display === 'block') {
                    contextMenu.style.display = 'none';
                    event.preventDefault();
                }
            });
            
            // Context menu actions
            document.getElementById('ctx-add-node').addEventListener('click', () => {
                objects.raycaster.setFromCamera(state.mousePosition, objects.camera);
                const intersects = objects.raycaster.intersectObject(objects.cubeWireframe);
                
                if (intersects.length > 0) {
                    addNode(intersects[0].point);
                    updateStats();
                }
            });
            
            document.getElementById('ctx-clear-area').addEventListener('click', () => {
                objects.raycaster.setFromCamera(state.mousePosition, objects.camera);
                const intersects = objects.raycaster.intersectObject(objects.cubeWireframe);
                
                if (intersects.length > 0) {
                    const position = intersects[0].point;
                    const radius = 5;
                    
                    // Remove nodes within radius
                    for (let i = objects.nodes.length - 1; i >= 0; i--) {
                        const node = objects.nodes[i];
                        const distance = node.position.distanceTo(position);
                        
                        if (distance < radius) {
                            // Remove connections first
                            const connections = [...node.userData.connections]; // Create a copy to avoid mutation during iteration
                            connections.forEach(conn => {
                                objects.scene.remove(conn);
                                const connIndex = objects.connections.indexOf(conn);
                                if (connIndex > -1) {
                                    objects.connections.splice(connIndex, 1);
                                }
                            });
                            
                            // Remove node
                            objects.scene.remove(node);
                            objects.nodes.splice(i, 1);
                        }
                    }
                    
                    updateStats();
                    showToast(`Cleared nodes in radius ${radius}`);
                }
            });
            
            document.getElementById('ctx-explode-from-here').addEventListener('click', () => {
                objects.raycaster.setFromCamera(state.mousePosition, objects.camera);
                
                if (state.selectedNode) {
                    createExplosionEffect(state.selectedNode.position, 1.5);
                } else {
                    const intersects = objects.raycaster.intersectObject(objects.cubeWireframe);
                    
                    if (intersects.length > 0) {
                        createExplosionEffect(intersects[0].point, 1.0);
                    }
                }
            });
            
            document.getElementById('ctx-create-cluster').addEventListener('click', () => {
                objects.raycaster.setFromCamera(state.mousePosition, objects.camera);
                const intersects = objects.raycaster.intersectObject(objects.cubeWireframe);
                
                if (intersects.length > 0) {
                    const position = intersects[0].point;
                    const clusterSize = 5;
                    
                    // Create cluster of nodes
                    for (let i = 0; i < clusterSize; i++) {
                        const offset = new THREE.Vector3(
                            (Math.random() - 0.5) * 3,
                            (Math.random() - 0.5) * 3,
                            (Math.random() - 0.5) * 3
                        );
                        
                        addNode(position.clone().add(offset));
                    }
                    
                    updateStats();
                    showToast(`Created cluster of ${clusterSize} nodes`);
                }
            });
            
            // Setup UI control handlers
            setupUIControlHandlers();
        }
        
        /**
         * Setup all UI control handlers
         */
        function setupUIControlHandlers() {
            // Panel toggle
            document.getElementById('panel-toggle').addEventListener('click', () => {
                const panel = document.getElementById('control-panel');
                panel.classList.toggle('visible');
                ui.controlPanelVisible = panel.classList.contains('visible');
            });
            
            // Get all slider elements
            ui.sliders.size = document.getElementById('size-slider');
            ui.sliders.density = document.getElementById('density-slider');
            ui.sliders.energy = document.getElementById('energy-slider');
            ui.sliders.connection = document.getElementById('connection-slider');
            ui.sliders.rotation = document.getElementById('rotation-slider');
            
            // Get all value displays
            ui.values.size = document.getElementById('size-value');
            ui.values.density = document.getElementById('density-value');
            ui.values.energy = document.getElementById('energy-value');
            ui.values.connection = document.getElementById('connection-value');
            ui.values.rotation = document.getElementById('rotation-value');
            
            // Get buttons
            ui.buttons.reset = document.getElementById('reset-btn');
            ui.buttons.addNodes = document.getElementById('add-nodes-btn');
            ui.buttons.entangle = document.getElementById('entangle-btn');
            ui.buttons.explosion = document.getElementById('explosion-btn');
            ui.buttons.wireframe = document.getElementById('wireframe-btn');
            ui.buttons.glow = document.getElementById('glow-btn');
            ui.buttons.save = document.getElementById('save-btn');
            ui.buttons.fullscreen = document.getElementById('fullscreen-btn');
            
            // Initial button states
            ui.buttons.wireframe.classList.toggle('active', state.cube.wireframe);
            ui.buttons.glow.classList.toggle('active', state.nodes.glowEnabled);
            
            // Setup slider handlers
            ui.sliders.size.addEventListener('input', (event) => {
                state.cube.size = parseInt(event.target.value);
                ui.values.size.textContent = state.cube.size;
            });
            
            ui.sliders.size.addEventListener('change', () => {
                createQuantumCube();
            });
            
            ui.sliders.density.addEventListener('input', (event) => {
                state.cube.density = parseInt(event.target.value);
                ui.values.density.textContent = state.cube.density;
            });
            
            ui.sliders.density.addEventListener('change', () => {
                createQuantumCube();
            });
            
            ui.sliders.energy.addEventListener('input', (event) => {
                state.energy = parseInt(event.target.value);
                ui.values.energy.textContent = state.energy;
                
                // Update point light intensity
                if (objects.pointLight) {
                    objects.pointLight.intensity = state.energy / 50;
                }
                
                updateStats();
            });
            
            ui.sliders.connection.addEventListener('input', (event) => {
                state.nodes.connectionThreshold = parseInt(event.target.value);
                ui.values.connection.textContent = state.nodes.connectionThreshold;
            });
            
            ui.sliders.rotation.addEventListener('input', (event) => {
                state.cube.rotationSpeed = parseInt(event.target.value);
                ui.values.rotation.textContent = state.cube.rotationSpeed;
            });
            
            // Button handlers
            ui.buttons.reset.addEventListener('click', () => {
                createQuantumCube();
                showToast("System reset complete");
            });
            
            ui.buttons.addNodes.addEventListener('click', () => {
                // Add 10 more nodes at random positions
                const halfSize = state.cube.size / 2 * 0.8;
                
                for (let i = 0; i < 10; i++) {
                    const position = new THREE.Vector3(
                        (Math.random() * 2 - 1) * halfSize,
                        (Math.random() * 2 - 1) * halfSize,
                        (Math.random() * 2 - 1) * halfSize
                    );
                    
                    addNode(position);
                }
                
                updateStats();
                showToast("Added 10 quantum nodes");
            });
            
            ui.buttons.entangle.addEventListener('click', () => {
                entangleNodes();
            });
            
            ui.buttons.explosion.addEventListener('click', () => {
                createExplosionEffect(new THREE.Vector3(0, 0, 0), 2.0);
            });
            
            ui.buttons.wireframe.addEventListener('click', () => {
                state.cube.wireframe = !state.cube.wireframe;
                if (objects.cubeWireframe) {
                    objects.cubeWireframe.visible = state.cube.wireframe;
                }
                
                ui.buttons.wireframe.classList.toggle('active', state.cube.wireframe);
                showToast(`Wireframe ${state.cube.wireframe ? 'enabled' : 'disabled'}`);
            });
            
            ui.buttons.glow.addEventListener('click', () => {
                state.nodes.glowEnabled = !state.nodes.glowEnabled;
                
                objects.nodes.forEach(node => {
                    node.material.emissiveIntensity = state.nodes.glowEnabled ? 0.5 : 0;
                });
                
                ui.buttons.glow.classList.toggle('active', state.nodes.glowEnabled);
                showToast(`Node glow ${state.nodes.glowEnabled ? 'enabled' : 'disabled'}`);
            });
            
            ui.buttons.save.addEventListener('click', () => {
                try {
                    // Render the scene at high quality
                    objects.renderer.render(objects.scene, objects.camera);
                    
                    // Get the canvas data
                    const imgData = objects.renderer.domElement.toDataURL('image/png');
                    
                    // Create a download link
                    const link = document.createElement('a');
                    link.href = imgData;
                    link.download = `quantum-visualization-${Date.now()}.png`;
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                    
                    showToast("Visualization image saved");
                } catch (error) {
                    console.error("Error saving image:", error);
                    showToast("Error saving image", "error");
                }
            });
            
            ui.buttons.fullscreen.addEventListener('click', () => {
                if (!document.fullscreenElement) {
                    document.documentElement.requestFullscreen().catch((err) => {
                        showToast("Fullscreen error: " + err.message, "error");
                    });
                } else {
                    document.exitFullscreen();
                }
            });
            
            // Color selection boxes
            const colorBoxes = document.querySelectorAll('.colorbox');
            
            colorBoxes.forEach(box => {
                ui.colorBoxes.push(box);
                
                box.addEventListener('click', () => {
                    // Remove active class from all boxes
                    ui.colorBoxes.forEach(b => b.classList.remove('active'));
                    
                    // Add active class to clicked box
                    box.classList.add('active');
                    
                    // Update node color
                    const colorHex = box.getAttribute('data-color');
                    const colorInt = hexToInt(colorHex);
                    
                    state.nodes.color = colorInt;
                    
                    // Update existing nodes
                    objects.nodes.forEach(node => {
                        node.material.color.set(colorInt);
                        node.material.emissive.set(colorInt);
                        node.userData.originalColor = colorInt;
                    });
                });
            });
            
            // Modal close handlers
            document.querySelector('.modal-close').addEventListener('click', () => {
                document.getElementById('modal').classList.remove('visible');
            });
            
            document.getElementById('modal-cancel').addEventListener('click', () => {
                document.getElementById('modal').classList.remove('visible');
            });
            
            document.getElementById('modal-confirm').addEventListener('click', () => {
                document.getElementById('modal').classList.remove('visible');
                
                if (ui.modalCallback) {
                    ui.modalCallback();
                    ui.modalCallback = null;
                }
            });
        }
        
        /**
         * Show the modal dialog with custom content
         */
        function showModal(title, content, confirmCallback = null) {
            document.querySelector('.modal-title').textContent = title;
            document.getElementById('modal-body').innerHTML = content;
            document.getElementById('modal').classList.add('visible');
            
            ui.modalCallback = confirmCallback;
        }
        
        /**
         * Simulated loading sequence
         */
        function simulateLoading() {
            const progressBar = document.getElementById('progress-fill');
            const loadingText = document.getElementById('loading-text');
            
            const loadingMessages = [
                "Loading core components...",
                "Initializing quantum matrix...",
                "Calibrating visualization engine...",
                "Preparing node configurations...",
                "Connecting entanglement system...",
                "Loading particle physics...",
                "Initializing UI elements...",
                "Optimizing rendering pipeline...",
                "Final calibration..."
            ];
            
            let progress = 0;
            
            const interval = setInterval(() => {
                progress += Math.random() * 8;
                
                if (progress >= 100) {
                    progress = 100;
                    clearInterval(interval);
                    
                    setTimeout(() => {
                        document.getElementById('loading').style.opacity = 0;
                        
                        setTimeout(() => {
                            document.getElementById('loading').style.display = 'none';
                            
                            // Initialize visualization after loading
                            initVisualization();
                        }, 500);
                    }, 500);
                }
                
                progressBar.style.width = `${progress}%`;
                
                // Update loading message
                const messageIndex = Math.floor((progress / 100) * loadingMessages.length);
                loadingText.textContent = loadingMessages[Math.min(messageIndex, loadingMessages.length - 1)];
            }, 200);
        }
        
        // Start loading sequence
        window.addEventListener('load', () => {
            simulateLoading();
            
            // Set initial UI values
            document.getElementById('size-value').textContent = state.cube.size;
            document.getElementById('density-value').textContent = state.cube.density;
            document.getElementById('energy-value').textContent = state.energy;
            document.getElementById('connection-value').textContent = state.nodes.connectionThreshold;
            document.getElementById('rotation-value').textContent = state.cube.rotationSpeed;
        });
    })();
    </script>
</body>
</html>
