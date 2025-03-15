import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { ArrowLeft, Brain, Download, Maximize2, Play, Rotate3D, Save, ZoomIn } from 'lucide-react';

const MoleculeViewer = () => {
  const [simulationRunning, setSimulationRunning] = useState(false);
  const [simulationProgress, setSimulationProgress] = useState(0);
  const [quantumStability, setQuantumStability] = useState(82);
  const [activeTab, setActiveTab] = useState('binding');
  const [viewMode, setViewMode] = useState('3d');
  const [simulationLog, setSimulationLog] = useState([
    { time: '14:32:05', message: 'Quantum core initialized' },
    { time: '14:32:08', message: 'Loading molecular structure data' },
    { time: '14:32:12', message: 'Structure loaded. Ready for simulation.' }
  ]);
  
  // Mock binding site data
  const [bindingSites] = useState([
    { site: 'Site A', energy: -8.7, probability: 68, stability: 72 },
    { site: 'Site B', energy: -7.2, probability: 52, stability: 81 },
    { site: 'Site C', energy: -9.3, probability: 89, stability: 76 },
    { site: 'Site D', energy: -6.8, probability: 45, stability: 63 },
    { site: 'Site E', energy: -8.1, probability: 59, stability: 79 }
  ]);
  
  // Mock simulation results over time
  const [simulationResults] = useState([
    { step: 0, energyA: -5.2, energyB: -4.8, energyC: -6.1, energyD: -4.2, energyE: -5.5 },
    { step: 10, energyA: -6.8, energyB: -5.3, energyC: -7.2, energyD: -4.6, energyE: -6.2 },
    { step: 20, energyA: -7.5, energyB: -6.1, energyC: -8.4, energyD: -5.1, energyE: -7.0 },
    { step: 30, energyA: -8.1, energyB: -6.7, energyC: -8.9, energyD: -5.9, energyE: -7.6 },
    { step: 40, energyA: -8.5, energyB: -7.0, energyC: -9.2, energyD: -6.5, energyE: -7.9 },
    { step: 50, energyA: -8.7, energyB: -7.2, energyC: -9.3, energyD: -6.8, energyE: -8.1 }
  ]);
  
  const startSimulation = () => {
    setSimulationRunning(true);
    setSimulationProgress(0);
    addToLog('Starting quantum-enhanced molecular simulation');
    
    // Simulate progress updates
    const interval = setInterval(() => {
      setSimulationProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          setSimulationRunning(false);
          addToLog('Simulation complete. Results ready for analysis.');
          return 100;
        }
        
        // Add random log messages during simulation
        if (prev === 20) addToLog('Initial binding configurations identified');
        if (prev === 40) addToLog('Quantum fluctuations stabilizing around site C');
        if (prev === 60) addToLog('Secondary binding site detected with 52% probability');
        if (prev === 80) addToLog('Final energy calculations in progress');
        
        return prev + 5;
      });
      
      // Random fluctuation in quantum stability
      setQuantumStability(prev => Math.max(70, Math.min(95, prev + (Math.random() * 4 - 2))));
    }, 500);
  };
  
  const addToLog = (message) => {
    const now = new Date();
    const timeString = now.toTimeString().split(' ')[0];
    setSimulationLog(prev => [...prev, { time: timeString, message }]);
  };
  
  // Auto-scroll log to bottom when new entries are added
  useEffect(() => {
    const logContainer = document.getElementById('simulation-log');
    if (logContainer) {
      logContainer.scrollTop = logContainer.scrollHeight;
    }
  }, [simulationLog]);

  return (
    <div className="flex flex-col h-screen bg-gray-100">
      {/* Header */}
      <header className="bg-indigo-800 text-white p-4">
        <div className="flex justify-between items-center">
          <div className="flex items-center space-x-2">
            <ArrowLeft size={20} className="cursor-pointer hover:text-indigo-300" />
            <div className="h-6 w-px bg-indigo-600 mx-2"></div>
            <Brain size={24} />
            <h1 className="text-xl font-bold">Kaleidoscope Molecular Simulation</h1>
          </div>
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <span>Quantum Stability:</span>
              <div className="w-24 h-3 bg-gray-700 rounded">
                <div 
                  className={`h-full rounded ${quantumStability > 85 ? 'bg-green-500' : 'bg-yellow-500'}`} 
                  style={{ width: `${quantumStability}%` }}
                ></div>
              </div>
              <span>{quantumStability}%</span>
            </div>
            <div className="flex items-center space-x-2">
              <button className="p-1 rounded bg-indigo-700 hover:bg-indigo-600">
                <Save size={16} />
              </button>
              <button className="p-1 rounded bg-indigo-700 hover:bg-indigo-600">
                <Download size={16} />
              </button>
            </div>
          </div>
        </div>
      </header>
      
      {/* Main Content */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left side - 3D visualization */}
        <div className="w-1/2 p-6 flex flex-col">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-semibold">Molecule Visualization</h2>
            <div className="flex space-x-2">
              <button 
                className={`px-3 py-1 rounded text-sm ${viewMode === '3d' ? 'bg-indigo-600 text-white' : 'bg-gray-200'}`}
                onClick={() => setViewMode('3d')}
              >
                3D View
              </button>
              <button 
                className={`px-3 py-1 rounded text-sm ${viewMode === '2d' ? 'bg-indigo-600 text-white' : 'bg-gray-200'}`}
                onClick={() => setViewMode('2d')}
              >
                2D Structure
              </button>
              <button 
                className={`px-3 py-1 rounded text-sm ${viewMode === 'surface' ? 'bg-indigo-600 text-white' : 'bg-gray-200'}`}
                onClick={() => setViewMode('surface')}
              >
                Surface
              </button>
            </div>
          </div>
          
          {/* 3D View Area */}
          <div className="flex-1 bg-white rounded shadow overflow-hidden relative">
            <div className="absolute top-4 right-4 flex flex-col space-y-2">
              <button className="p-2 rounded bg-white shadow hover:bg-gray-100">
                <ZoomIn size={18} />
              </button>
              <button className="p-2 rounded bg-white shadow hover:bg-gray-100">
                <Rotate3D size={18} />
              </button>
              <button className="p-2 rounded bg-white shadow hover:bg-gray-100">
                <Maximize2 size={18} />
              </button>
            </div>
            
            {/* Placeholder for 3D visualization */}
            <div className="h-full flex items-center justify-center bg-gradient-to-br from-gray-900 to-gray-800">
              <div className="text-center text-white">
                <div className="mb-4 text-lg">Interactive Molecular Structure</div>
                <div className="bg-indigo-900 bg-opacity-50 p-8 rounded-full inline-block">
                  <div className="relative">
                    {/* Simulate simplified molecule visualization with basic shapes */}
                    <div className="absolute w-6 h-6 rounded-full bg-red-500 shadow-lg" style={{ top: -40, left: -20 }}></div>
                    <div className="absolute w-4 h-4 rounded-full bg-blue-500 shadow-lg" style={{ top: -20, left: 20 }}></div>
                    <div className="absolute w-5 h-5 rounded-full bg-green-500 shadow-lg" style={{ top: 10, left: -30 }}></div>
                    <div className="absolute w-4 h-4 rounded-full bg-yellow-500 shadow-lg" style={{ top: 30, left: 10 }}></div>
                    <div className="absolute w-3 h-3 rounded-full bg-purple-500 shadow-lg" style={{ top: 0, left: 0 }}></div>
                    
                    {/* Bonds */}
                    <div className="absolute w-16 h-1 bg-gray-300 rotate-45" style={{ top: -27, left: -15 }}></div>
                    <div className="absolute w-20 h-1 bg-gray-300 rotate-12" style={{ top: -32, left: -12 }}></div>
                    <div className="absolute w-14 h-1 bg-gray-300 rotate-90" style={{ top: -5, left: -28 }}></div>
                    <div className="absolute w-18 h-1 bg-gray-300 rotate-120" style={{ top: 15, left: -10 }}></div>
                    
                    <div className="w-20 h-20 flex items-center justify-center">
                      <span className="text-white text-xs">Site C</span>
                    </div>
                  </div>
                </div>
                <div className="mt-4 text-xs text-gray-400">
                  {viewMode === '3d' && 'Interactive 3D View - Highlighted: Binding Site C'}
                  {viewMode === '2d' && '2D Structure View - Highlighted: Binding Site C'}
                  {viewMode === 'surface' && 'Surface Potential View - Highlighted: Binding Site C'}
                </div>
              </div>
            </div>
            
            {/* Simulation controls */}
            <div className="bg-gray-100 p-3 border-t border-gray-200">
              <div className="flex justify-between items-center">
                <div className="flex items-center space-x-3">
                  <button 
                    className={`px-4 py-2 rounded flex items-center space-x-2 ${
                      simulationRunning 
                        ? 'bg-yellow-500 hover:bg-yellow-600 text-white' 
                        : 'bg-indigo-600 hover:bg-indigo-700 text-white'
                    }`}
                    onClick={startSimulation}
                    disabled={simulationRunning}
                  >
                    <Play size={16} />
                    <span>{simulationRunning ? 'Simulating...' : 'Run Simulation'}</span>
                  </button>
                </div>
                
                <div className="flex items-center space-x-2 flex-1 ml-4">
                  <div className="text-xs">{simulationProgress}%</div>
                  <div className="flex-1 h-2 bg-gray-300 rounded">
                    <div 
                      className="h-full bg-indigo-600 rounded" 
                      style={{ width: `${simulationProgress}%` }}
                    ></div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        {/* Right side - Analysis and Logs */}
        <div className="w-1/2 p-6 flex flex-col">
          <div className="bg-white rounded shadow mb-6 flex-1">
            <div className="border-b border-gray-200">
              <div className="flex">
                <button 
                  className={`px-4 py-3 font-medium ${activeTab === 'binding' ? 'border-b-2 border-indigo-600 text-indigo-600' : 'text-gray-500 hover:text-gray-700'}`}
                  onClick={() => setActiveTab('binding')}
                >
                  Binding Sites
                </button>
                <button 
                  className={`px-4 py-3 font-medium ${activeTab === 'energy' ? 'border-b-2 border-indigo-600 text-indigo-600' : 'text-gray-500 hover:text-gray-700'}`}
                  onClick={() => setActiveTab('energy')}
                >
                  Energy Profile
                </button>
                <button 
                  className={`px-4 py-3 font-medium ${activeTab === 'quantum' ? 'border-b-2 border-indigo-600 text-indigo-600' : 'text-gray-500 hover:text-gray-700'}`}
                  onClick={() => setActiveTab('quantum')}
                >
                  Quantum Effects
                </button>
              </div>
            </div>
            
            <div className="p-4">
              {activeTab === 'binding' && (
                <div>
                  <h3 className="text-lg font-medium mb-4">Binding Site Analysis</h3>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={bindingSites}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="site" />
                        <YAxis yAxisId="left" orientation="left" stroke="#8884d8" />
                        <YAxis yAxisId="right" orientation="right" stroke="#82ca9d" />
                        <Tooltip />
                        <Legend />
                        <Bar yAxisId="left" dataKey="probability" name="Binding Probability (%)" fill="#8884d8" />
                        <Bar yAxisId="right" dataKey="stability" name="Stability Index" fill="#82ca9d" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="mt-4 p-4 bg-gray-50 rounded border border-gray-200">
                    <h4 className="font-medium mb-2">Binding Analysis Summary</h4>
                    <p className="text-sm text-gray-700">
                      Quantum simulation results indicate <strong>Site C</strong> has the highest binding probability (89%) 
                      with an energy score of <strong>-9.3 kcal/mol</strong>. This binding configuration demonstrates 
                      stable hydrogen bonding and favorable π-π stacking interactions with the receptor pocket.
                    </p>
                    <p className="text-sm text-gray-700 mt-2">
                      Secondary binding at <strong>Site A</strong> (68% probability) may provide an alternative 
                      interaction pathway that could be exploited for drug design optimization.
                    </p>
                  </div>
                </div>
              )}
              
              {activeTab === 'energy' && (
                <div>
                  <h3 className="text-lg font-medium mb-4">Energy Profile Over Simulation</h3>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={simulationResults}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="step" label={{ value: 'Simulation Step', position: 'insideBottomRight', offset: -10 }} />
                        <YAxis label={{ value: 'Binding Energy (kcal/mol)', angle: -90, position: 'insideLeft' }} />
                        <Tooltip />
                        <Legend />
                        <Line type="monotone" dataKey="energyA" name="Site A" stroke="#8884d8" />
                        <Line type="monotone" dataKey="energyB" name="Site B" stroke="#82ca9d" />
                        <Line type="monotone" dataKey="energyC" name="Site C" stroke="#ff7300" activeDot={{ r: 8 }} />
                        <Line type="monotone" dataKey="energyD" name="Site D" stroke="#0088fe" />
                        <Line type="monotone" dataKey="energyE" name="Site E" stroke="#ff8042" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="mt-4 p-4 bg-gray-50 rounded border border-gray-200">
                    <h4 className="font-medium mb-2">Energy Convergence Analysis</h4>
                    <p className="text-sm text-gray-700">
                      Energy profiles have converged for all binding sites, with Site C showing the lowest
                      final energy state at -9.3 kcal/mol. Quantum tunneling effects helped overcome energy
                      barriers at steps 20-30, resulting in more accurate binding predictions than 
                      conventional molecular dynamics would achieve.
                    </p>
                  </div>
                </div>
              )}
              
              {activeTab === 'quantum' && (
                <div>
                  <h3 className="text-lg font-medium mb-4">Quantum Effects Analysis</h3>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-gray-50 p-3 rounded border border-gray-200">
                      <h4 className="font-medium mb-2 text-sm">Quantum Tunneling</h4>
                      <div className="text-sm text-gray-700">
                        Enhanced barrier crossing observed in binding pocket.
                        <div className="mt-2 w-full h-2 bg-gray-200 rounded">
                          <div className="h-full bg-purple-500 rounded" style={{ width: '84%' }}></div>
                        </div>
                        <div className="text-xs mt-1 text-right">84% contribution</div>
                      </div>
                    </div>
                    <div className="bg-gray-50 p-3 rounded border border-gray-200">
                      <h4 className="font-medium mb-2 text-sm">State Superposition</h4>
                      <div className="text-sm text-gray-700">
                        Molecular conformations explored simultaneously.
                        <div className="mt-2 w-full h-2 bg-gray-200 rounded">
                          <div className="h-full bg-blue-500 rounded" style={{ width: '76%' }}></div>
                        </div>
                        <div className="text-xs mt-1 text-right">76% contribution</div>
                      </div>
                    </div>
                    <div className="bg-gray-50 p-3 rounded border border-gray-200">
                      <h4 className="font-medium mb-2 text-sm">Entanglement Effects</h4>
                      <div className="text-sm text-gray-700">
                        Correlated movements between binding site residues.
                        <div className="mt-2 w-full h-2 bg-gray-200 rounded">
                          <div className="h-full bg-green-500 rounded" style={{ width: '92%' }}></div>
                        </div>
                        <div className="text-xs mt-1 text-right">92% contribution</div>
                      </div>
                    </div>
                    <div className="bg-gray-50 p-3 rounded border border-gray-200">
                      <h4 className="font-medium mb-2 text-sm">Quantum Coherence</h4>
                      <div className="text-sm text-gray-700">
                        Maintained during critical binding phase.
                        <div className="mt-2 w-full h-2 bg-gray-200 rounded">
                          <div className="h-full bg-red-500 rounded" style={{ width: '68%' }}></div>
                        </div>
                        <div className="text-xs mt-1 text-right">68% contribution</div>
                      </div>
                    </div>
                  </div>
                  <div className="mt-4 p-4 bg-gray-50 rounded border border-gray-200">
                    <h4 className="font-medium mb-2">Quantum Simulation Advantage</h4>
                    <p className="text-sm text-gray-700">
                      The quantum-inspired algorithms have identified binding modes that would be
                      missed by classical approaches. Entanglement effects between key residues
                      enhance prediction accuracy by ~43% compared to conventional methods.
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
          
          {/* Log output */}
          <div className="bg-white rounded shadow flex-none h-64">
            <div className="border-b border-gray-200 px-4 py-2 bg-gray-50 flex justify-between items-center">
              <h3 className="font-medium">Simulation Log</h3>
              <div className="text-xs text-gray-500">
                {simulationRunning ? 'Simulation in progress...' : 'Awaiting commands'}
              </div>
            </div>
            <div id="simulation-log" className="p-2 h-full overflow-y-auto text-sm font-mono">
              {simulationLog.map((entry, index) => (
                <div key={index} className="mb-1">
                  <span className="text-gray-500">[{entry.time}]</span>{' '}
                  <span className="text-gray-800">{entry.message}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MoleculeViewer;
