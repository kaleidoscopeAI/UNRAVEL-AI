import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { AlertCircle, Brain, Database, FileText, Grid, LayoutDashboard, MoreHorizontal, Plus, RefreshCcw, Search } from 'lucide-react';

const KaleidoscopeDashboard = () => {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [insightCount, setInsightCount] = useState(0);
  const [nodeCount, setNodeCount] = useState(0);
  const [systemStatus, setSystemStatus] = useState('Active');
  const [cubeStability, setCubeStability] = useState(92);
  const [selectedPerspective, setSelectedPerspective] = useState('validated');
  const [isProcessing, setIsProcessing] = useState(false);
  const [chatInput, setChatInput] = useState('');
  const [chatHistory, setChatHistory] = useState([
    { sender: 'system', message: 'Kaleidoscope AI system initialized. How may I assist you today?' }
  ]);
  
  // Simulate activity data
  const [systemActivity] = useState([
    { time: '08:00', nodes: 42, insights: 18, stability: 89 },
    { time: '09:00', nodes: 58, insights: 27, stability: 86 },
    { time: '10:00', nodes: 75, insights: 36, stability: 90 },
    { time: '11:00', nodes: 81, insights: 42, stability: 92 },
    { time: '12:00', nodes: 95, insights: 51, stability: 88 },
    { time: '13:00', nodes: 112, insights: 63, stability: 91 },
    { time: '14:00', nodes: 128, insights: 72, stability: 94 },
    { time: '15:00', nodes: 137, insights: 81, stability: 93 },
  ]);

  // Mock data for visualization
  const [nodeNetwork] = useState({
    nodes: [
      { id: 'n1', type: 'data', size: 25 },
      { id: 'n2', type: 'insight', size: 15 },
      { id: 'n3', type: 'data', size: 20 },
      { id: 'n4', type: 'super', size: 30 },
      { id: 'n5', type: 'insight', size: 18 },
      { id: 'n6', type: 'data', size: 22 },
      { id: 'n7', type: 'super', size: 35 },
    ],
    links: [
      { source: 'n1', target: 'n2', strength: 0.7 },
      { source: 'n1', target: 'n4', strength: 0.5 },
      { source: 'n2', target: 'n4', strength: 0.8 },
      { source: 'n3', target: 'n4', strength: 0.6 },
      { source: 'n3', target: 'n5', strength: 0.9 },
      { source: 'n5', target: 'n7', strength: 0.7 },
      { source: 'n6', target: 'n7', strength: 0.8 },
    ]
  });
  
  // Simulate the increasing number of insights and nodes
  useEffect(() => {
    const timer = setInterval(() => {
      setInsightCount(prev => prev + Math.floor(Math.random() * 3));
      setNodeCount(prev => prev + Math.floor(Math.random() * 5));
      setCubeStability(prev => Math.min(100, Math.max(80, prev + (Math.random() * 2 - 1))));
    }, 5000);
    
    return () => clearInterval(timer);
  }, []);
  
  const handleSendMessage = () => {
    if (!chatInput.trim()) return;
    
    // Add user message to chat
    setChatHistory(prev => [...prev, { sender: 'user', message: chatInput }]);
    
    // Simulate processing
    setIsProcessing(true);
    
    // Add system response after a delay (simulating processing time)
    setTimeout(() => {
      let responseMessage = '';
      
      // Generate different responses based on the perspective
      if (selectedPerspective === 'validated') {
        responseMessage = `Based on validated pattern analysis, I can confirm that ${chatInput.includes('market') ? 'market trends indicate a 73% probability of sector growth in Q3' : 'the data shows significant correlation patterns between the variables you mentioned'}. This insight has been validated across multiple nodes with 92% confidence.`;
      } else if (selectedPerspective === 'speculative') {
        responseMessage = `From a speculative perspective, I'd suggest considering that ${chatInput.includes('market') ? 'while conventional analysis shows market growth, there's an intriguing possibility of disruption from emerging technologies not yet fully factored into market projections' : 'the relationship between these variables might be influenced by an external factor we haven't considered yet'}. This represents an alternative view with 63% confidence.`;
      } else {
        responseMessage = `The quantum-perspective analysis reveals a superposition of possibilities: ${chatInput.includes('market') ? 'the market exhibits both trending patterns and counter-pattern indicators simultaneously, suggesting we're at a decision point that could collapse toward either outcome' : 'the system detects multiple valid interpretations of this data that exist in parallel until more information becomes available'}. This quantum state has not yet collapsed to a definitive answer.`;
      }
      
      // Add system response
      setChatHistory(prev => [...prev, { sender: 'system', message: responseMessage }]);
      setIsProcessing(false);
    }, 1500);
    
    // Clear input
    setChatInput('');
  };

  return (
    <div className="flex flex-col h-screen bg-gray-100">
      {/* Header */}
      <header className="bg-indigo-800 text-white p-4">
        <div className="flex justify-between items-center">
          <div className="flex items-center space-x-2">
            <Brain size={24} />
            <h1 className="text-xl font-bold">Kaleidoscope AI System</h1>
          </div>
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <AlertCircle size={16} className={systemStatus === 'Active' ? 'text-green-400' : 'text-yellow-400'} />
              <span>{systemStatus}</span>
            </div>
            <div className="flex items-center space-x-2">
              <span>Cube Stability:</span>
              <div className="w-24 h-3 bg-gray-700 rounded">
                <div 
                  className={`h-full rounded ${cubeStability > 90 ? 'bg-green-500' : cubeStability > 80 ? 'bg-yellow-500' : 'bg-red-500'}`} 
                  style={{ width: `${cubeStability}%` }}
                ></div>
              </div>
              <span>{cubeStability}%</span>
            </div>
          </div>
        </div>
      </header>
      
      {/* Main Content */}
      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <aside className="w-16 bg-indigo-900 text-white flex flex-col items-center py-4">
          <button 
            className={`p-2 rounded mb-4 ${activeTab === 'dashboard' ? 'bg-indigo-700' : 'hover:bg-indigo-800'}`}
            onClick={() => setActiveTab('dashboard')}
          >
            <LayoutDashboard size={20} />
          </button>
          <button 
            className={`p-2 rounded mb-4 ${activeTab === 'nodes' ? 'bg-indigo-700' : 'hover:bg-indigo-800'}`}
            onClick={() => setActiveTab('nodes')}
          >
            <Grid size={20} />
          </button>
          <button 
            className={`p-2 rounded mb-4 ${activeTab === 'data' ? 'bg-indigo-700' : 'hover:bg-indigo-800'}`}
            onClick={() => setActiveTab('data')}
          >
            <Database size={20} />
          </button>
          <button 
            className={`p-2 rounded mb-4 ${activeTab === 'search' ? 'bg-indigo-700' : 'hover:bg-indigo-800'}`}
            onClick={() => setActiveTab('search')}
          >
            <Search size={20} />
          </button>
          <button 
            className={`p-2 rounded ${activeTab === 'reports' ? 'bg-indigo-700' : 'hover:bg-indigo-800'}`}
            onClick={() => setActiveTab('reports')}
          >
            <FileText size={20} />
          </button>
        </aside>
        
        {/* Main content area */}
        <main className="flex-1 overflow-hidden flex">
          {/* Left panel */}
          <div className="w-2/3 p-6 overflow-auto">
            <h2 className="text-2xl font-semibold mb-6">System Overview</h2>
            
            {/* Stats */}
            <div className="grid grid-cols-3 gap-6 mb-6">
              <div className="bg-white p-4 rounded shadow">
                <h3 className="text-lg font-medium text-gray-500">Active Nodes</h3>
                <p className="text-3xl font-bold">{nodeCount}</p>
                <div className="mt-2 text-sm text-green-500">+12% from last hour</div>
              </div>
              <div className="bg-white p-4 rounded shadow">
                <h3 className="text-lg font-medium text-gray-500">Validated Insights</h3>
                <p className="text-3xl font-bold">{insightCount}</p>
                <div className="mt-2 text-sm text-green-500">+8% from last hour</div>
              </div>
              <div className="bg-white p-4 rounded shadow">
                <h3 className="text-lg font-medium text-gray-500">SuperNodes Formed</h3>
                <p className="text-3xl font-bold">{Math.floor(nodeCount / 15)}</p>
                <div className="mt-2 text-sm text-blue-500">2 new formations</div>
              </div>
            </div>
            
            {/* Activity Chart */}
            <div className="bg-white p-4 rounded shadow mb-6">
              <h3 className="text-lg font-medium mb-4">System Activity</h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={systemActivity}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" />
                    <YAxis yAxisId="left" />
                    <YAxis yAxisId="right" orientation="right" />
                    <Tooltip />
                    <Legend />
                    <Line 
                      yAxisId="left"
                      type="monotone" 
                      dataKey="nodes" 
                      stroke="#8884d8" 
                      name="Active Nodes"
                    />
                    <Line 
                      yAxisId="left"
                      type="monotone" 
                      dataKey="insights" 
                      stroke="#82ca9d" 
                      name="New Insights"
                    />
                    <Line 
                      yAxisId="right"
                      type="monotone" 
                      dataKey="stability" 
                      stroke="#ff7300" 
                      name="Cube Stability %"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
            
            {/* Visualization placeholder */}
            <div className="bg-white p-4 rounded shadow">
              <h3 className="text-lg font-medium mb-4">Knowledge Cube Visualization</h3>
              <div className="border-2 border-dashed border-gray-300 rounded h-64 flex items-center justify-center">
                <div className="text-center">
                  <div className="text-gray-500 mb-2">3D Cube Visualization</div>
                  <button className="bg-indigo-600 text-white px-4 py-2 rounded hover:bg-indigo-700">
                    Open Full View
                  </button>
                </div>
              </div>
            </div>
          </div>
          
          {/* Right panel - Jacob Chatbot */}
          <div className="w-1/3 border-l border-gray-200 flex flex-col">
            <div className="p-4 border-b border-gray-200 bg-white">
              <div className="flex justify-between items-center">
                <h2 className="text-lg font-semibold">Jacob - Cognitive Interface</h2>
                <div className="flex space-x-1">
                  <button className="p-1 hover:bg-gray-100 rounded">
                    <RefreshCcw size={16} />
                  </button>
                  <button className="p-1 hover:bg-gray-100 rounded">
                    <MoreHorizontal size={16} />
                  </button>
                </div>
              </div>
              
              {/* Perspective selector */}
              <div className="mt-2 flex border rounded overflow-hidden">
                <button 
                  className={`flex-1 py-1 px-2 text-sm ${selectedPerspective === 'validated' ? 'bg-indigo-600 text-white' : 'bg-gray-100'}`}
                  onClick={() => setSelectedPerspective('validated')}
                >
                  Validated
                </button>
                <button 
                  className={`flex-1 py-1 px-2 text-sm ${selectedPerspective === 'speculative' ? 'bg-indigo-600 text-white' : 'bg-gray-100'}`}
                  onClick={() => setSelectedPerspective('speculative')}
                >
                  Speculative
                </button>
                <button 
                  className={`flex-1 py-1 px-2 text-sm ${selectedPerspective === 'quantum' ? 'bg-indigo-600 text-white' : 'bg-gray-100'}`}
                  onClick={() => setSelectedPerspective('quantum')}
                >
                  Quantum
                </button>
              </div>
            </div>
            
            {/* Chat messages */}
            <div className="flex-1 overflow-y-auto p-4 bg-gray-50">
              {chatHistory.map((item, index) => (
                <div key={index} className={`mb-4 ${item.sender === 'user' ? 'text-right' : ''}`}>
                  <div 
                    className={`inline-block p-3 rounded-lg max-w-xs lg:max-w-md ${
                      item.sender === 'user' 
                        ? 'bg-indigo-600 text-white' 
                        : 'bg-white text-gray-800 border border-gray-200'
                    }`}
                  >
                    {item.message}
                  </div>
                </div>
              ))}
              {isProcessing && (
                <div className="mb-4">
                  <div className="inline-block p-3 rounded-lg bg-white text-gray-800 border border-gray-200">
                    <div className="flex space-x-2">
                      <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: '0ms' }}></div>
                      <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: '150ms' }}></div>
                      <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: '300ms' }}></div>
                    </div>
                  </div>
                </div>
              )}
            </div>
            
            {/* Chat input */}
            <div className="p-4 border-t border-gray-200 bg-white">
              <div className="flex">
                <input
                  type="text"
                  className="flex-1 border border-gray-300 rounded-l p-2"
                  placeholder="Ask Jacob a question..."
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                />
                <button 
                  className="bg-indigo-600 text-white px-4 rounded-r hover:bg-indigo-700"
                  onClick={handleSendMessage}
                >
                  <Plus size={20} />
                </button>
              </div>
              <div className="mt-2 text-xs text-gray-500">
                {selectedPerspective === 'validated' && "Using Kaleidoscope Engine for validated insights"}
                {selectedPerspective === 'speculative' && "Using Perspective Engine for speculative exploration"}
                {selectedPerspective === 'quantum' && "Using Quantum Core for probabilistic reasoning"}
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
};

export default KaleidoscopeDashboard;
