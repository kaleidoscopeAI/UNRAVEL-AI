#!/usr/bin/env python3
"""
Kaleidoscope AI Visualization Module
Provides visualization tools for the Kaleidoscope AI system.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import json
import os
import io
import base64
from typing import Dict, List, Any, Optional, Tuple, Union

class KaleidoscopeVisualizer:
    """
    Visualization tools for the Kaleidoscope AI system.
    """
    
    def __init__(self, output_dir: str = './visualizations'):
        """Initialize visualizer with output directory."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Define color schemes
        self.node_colors = {
            'software_analysis': '#ff7f0e',    # Orange
            'molecular_modeling': '#2ca02c',   # Green
            'pattern_detection': '#d62728',    # Red
            'knowledge_integration': '#9467bd' # Purple
        }
        
        # Default color for unknown domain
        self.default_color = '#1f77b4'  # Blue
        
        # Colorscale for heatmaps
        self.colorscale = [
            [0.0, '#f7fbff'],
            [0.2, '#deebf7'],
            [0.4, '#9ecae1'],
            [0.6, '#6baed6'],
            [0.8, '#3182bd'],
            [1.0, '#08519c']
        ]
    
    def visualize_cube(self, 
                     cube_data: Dict,
                     output_file: str = None,
                     interactive: bool = True) -> Union[str, None]:
        """
        Visualize the Kaleidoscope cube structure.
        
        Args:
            cube_data: Cube visualization data from the API
            output_file: Path to save the visualization (optional)
            interactive: Whether to create an interactive visualization
        
        Returns:
            Path to output file if saved, None otherwise
        """
        if interactive:
            return self._plotly_cube_visualization(cube_data, output_file)
        else:
            return self._matplotlib_cube_visualization(cube_data, output_file)
    
    def _plotly_cube_visualization(self, 
                                cube_data: Dict,
                                output_file: str = None) -> str:
        """Create an interactive cube visualization using Plotly."""
        nodes = cube_data.get('nodes', {})
        connections = cube_data.get('connections', [])
        dimensions = cube_data.get('dimensions', (10, 10, 10))
        
        # Prepare node data
        node_x, node_y, node_z = [], [], []
        node_colors, node_sizes, node_texts = [], [], []
        
        for node_id, node_data in nodes.items():
            position = node_data.get('position', (0, 0, 0))
            node_x.append(position[0])
            node_y.append(position[1])
            node_z.append(position[2])
            
            domain = node_data.get('domain', 'unknown')
            node_colors.append(self.node_colors.get(domain, self.default_color))
            
            # Node size based on connections
            connections_count = node_data.get('connections', 1)
            node_sizes.append(10 + connections_count * 2)
            
            # Node text for hover info
            node_texts.append(f"ID: {node_id}<br>Domain: {domain}<br>Connections: {connections_count}")
        
        # Create 3D scatter plot for nodes
        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=0.5, color='#ffffff')
            ),
            text=node_texts,
            hoverinfo='text'
        )
        
        # Prepare edge data
        edge_x, edge_y, edge_z = [], [], []
        
        for source_id, target_id, strength in connections:
            if source_id in nodes and target_id in nodes:
                source_pos = nodes[source_id].get('position', (0, 0, 0))
                target_pos = nodes[target_id].get('position', (0, 0, 0))
                
                # Add line between nodes
                edge_x.extend([source_pos[0], target_pos[0], None])
                edge_y.extend([source_pos[1], target_pos[1], None])
                edge_z.extend([source_pos[2], target_pos[2], None])
        
        # Create 3D line plot for edges
        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(width=1, color='rgba(100,100,100,0.4)'),
            hoverinfo='none'
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace])
        
        # Set layout
        fig.update_layout(
            title='Kaleidoscope AI Cube Structure',
            scene=dict(
                xaxis=dict(range=[0, dimensions[0]], title='X'),
                yaxis=dict(range=[0, dimensions[1]], title='Y'),
                zaxis=dict(range=[0, dimensions[2]], title='Z'),
                aspectratio=dict(x=1, y=1, z=1)
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            showlegend=False
        )
        
        # Add clusters visualization if available
        clusters = cube_data.get('clusters', {})
        if clusters:
            cluster_colors = px.colors.qualitative.Plotly
            cluster_traces = []
            
            for i, (cluster_id, cluster_data) in enumerate(clusters.items()):
                cluster_nodes = cluster_data.get('nodes', [])
                cluster_center = cluster_data.get('center', (0, 0, 0))
                
                # Add trace for cluster center
                cluster_traces.append(
                    go.Scatter3d(
                        x=[cluster_center[0]],
                        y=[cluster_center[1]],
                        z=[cluster_center[2]],
                        mode='markers',
                        marker=dict(
                            size=15,
                            color=cluster_colors[i % len(cluster_colors)],
                            symbol='diamond',
                            line=dict(width=1, color='#ffffff')
                        ),
                        text=f"Cluster: {cluster_id}<br>Size: {len(cluster_nodes)}",
                        hoverinfo='text',
                        name=f"Cluster {i+1}"
                    )
                )
            
            fig.add_traces(cluster_traces)
            fig.update_layout(showlegend=True)
        
        # Save to file if specified
        if output_file:
            output_path = os.path.join(self.output_dir, output_file)
            fig.write_html(output_path)
            return output_path
        
        # Otherwise return as base64 encoded HTML
        buffer = io.StringIO()
        fig.write_html(buffer)
        html_str = buffer.getvalue()
        
        encoded = base64.b64encode(html_str.encode()).decode()
        return encoded
    
    def _matplotlib_cube_visualization(self, 
                                    cube_data: Dict,
                                    output_file: str = None) -> str:
        """Create a static cube visualization using Matplotlib."""
        nodes = cube_data.get('nodes', {})
        connections = cube_data.get('connections', [])
        dimensions = cube_data.get('dimensions', (10, 10, 10))
        
        # Create figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot nodes
        for node_id, node_data in nodes.items():
            position = node_data.get('position', (0, 0, 0))
            domain = node_data.get('domain', 'unknown')
            connections_count = node_data.get('connections', 1)
            
            # Node size based on connections
            node_size = 20 + connections_count * 5
            
            # Node color based on domain
            node_color = self.node_colors.get(domain, self.default_color)
            
            ax.scatter(
                position[0], position[1], position[2],
                s=node_size, c=node_color, alpha=0.8,
                label=domain if domain not in ax.get_legend_handles_labels()[1] else ""
            )
            
            # Add node ID text
            ax.text(
                position[0], position[1], position[2],
                node_id.split('_')[1] if '_' in node_id else node_id,
                size=8, zorder=1, color='k'
            )
        
        # Plot connections
        for source_id, target_id, strength in connections:
            if source_id in nodes and target_id in nodes:
                source_pos = nodes[source_id].get('position', (0, 0, 0))
                target_pos = nodes[target_id].get('position', (0, 0, 0))
                
                # Line width based on connection strength
                line_width = 0.5 + strength * 2
                
                ax.plot(
                    [source_pos[0], target_pos[0]],
                    [source_pos[1], target_pos[1]],
                    [source_pos[2], target_pos[2]],
                    'gray', alpha=0.4, linewidth=line_width
                )
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(0, dimensions[0])
        ax.set_ylim(0, dimensions[1])
        ax.set_zlim(0, dimensions[2])
        ax.set_title('Kaleidoscope AI Cube Structure')
        
        # Add legend (only for unique domains)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), 
                title="Domains", loc="upper right")
        
        # Tight layout
        plt.tight_layout()
        
        # Save to file if specified
        if output_file:
            output_path = os.path.join(self.output_dir, output_file)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            return output_path
        
        # Otherwise return as base64 encoded PNG
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        buffer.seek(0)
        encoded = base64.b64encode(buffer.read()).decode()
        return encoded
    
    def visualize_network(self, 
                        network_data: Dict,
                        output_file: str = None,
                        interactive: bool = True) -> Union[str, None]:
        """
        Visualize the Kaleidoscope network structure.
        
        Args:
            network_data: Network visualization data from the API
            output_file: Path to save the visualization (optional)
            interactive: Whether to create an interactive visualization
        
        Returns:
            Path to output file if saved, None otherwise
        """
        if interactive:
            return self._plotly_network_visualization(network_data, output_file)
        else:
            return self._matplotlib_network_visualization(network_data, output_file)
    
    def _plotly_network_visualization(self, 
                                   network_data: Dict,
                                   output_file: str = None) -> str:
        """Create an interactive network visualization using Plotly."""
        nodes = network_data.get('nodes', [])
        edges = network_data.get('edges', [])
        
        # Create networkx graph
        G = nx.Graph()
        
        # Add nodes
        for node in nodes:
            G.add_node(
                node['id'],
                domain=node.get('domain', 'unknown'),
                position=node.get('position', [0, 0, 0])
            )
        
        # Add edges
        for edge in edges:
            G.add_edge(
                edge['source'],
                edge['target'],
                weight=edge.get('weight', 1.0)
            )
        
        # Get position layout
        pos = nx.spring_layout(G, seed=42)
        
        # Prepare node data
        node_x, node_y = [], []
        node_colors, node_sizes, node_texts = [], [], []
        
        for node_id, node_data in G.nodes(data=True):
            x, y = pos[node_id]
            node_x.append(x)
            node_y.append(y)
            
            domain = node_data.get('domain', 'unknown')
            node_colors.append(self.node_colors.get(domain, self.default_color))
            
            # Node size based on degree
            degree = G.degree(node_id)
            node_sizes.append(10 + degree * 2)
            
            # Node text for hover info
            node_texts.append(f"ID: {node_id}<br>Domain: {domain}<br>Connections: {degree}")
        
        # Create scatter plot for nodes
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=0.5, color='#ffffff')
            ),
            text=node_texts,
            hoverinfo='text'
        )
        
        # Prepare edge data
        edge_x, edge_y = [], []
        edge_colors = []
        
        for edge in G.edges(data=True):
            source, target = edge[0], edge[1]
            weight = edge[2].get('weight', 1.0)
            
            x0, y0 = pos[source]
            x1, y1 = pos[target]
            
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Edge color based on weight
            edge_colors.extend([weight, weight, weight])
        
        # Create line plot for edges
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(
                width=1,
                color=edge_colors,
                colorscale=self.colorscale,
                cmin=0,
                cmax=1
            ),
            hoverinfo='none'
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace])
        
        # Set layout
        fig.update_layout(
            title='Kaleidoscope AI Network Structure',
            showlegend=False,
            hovermode='closest',
            margin=dict(l=0, r=0, b=0, t=30),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        # Add domain legend
        unique_domains = set()
        for node_data in G.nodes(data=True):
            unique_domains.add(node_data[1].get('domain', 'unknown'))
        
        for i, domain in enumerate(unique_domains):
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(
                    size=10,
                    color=self.node_colors.get(domain, self.default_color)
                ),
                name=domain
            ))
        
        fig.update_layout(showlegend=True)
        
        # Save to file if specified
        if output_file:
            output_path = os.path.join(self.output_dir, output_file)
            fig.write_html(output_path)
            return output_path
        
        # Otherwise return as base64 encoded HTML
        buffer = io.StringIO()
        fig.write_html(buffer)
        html_str = buffer.getvalue()
        
        encoded = base64.b64encode(html_str.encode()).decode()
        return encoded
    
    def _matplotlib_network_visualization(self, 
                                       network_data: Dict,
                                       output_file: str = None) -> str:
        """Create a static network visualization using Matplotlib."""
        nodes = network_data.get('nodes', [])
        edges = network_data.get('edges', [])
        
        # Create networkx graph
        G = nx.Graph()
        
        # Add nodes
        for node in nodes:
            G.add_node(
                node['id'],
                domain=node.get('domain', 'unknown'),
                position=node.get('position', [0, 0, 0])
            )
        
        # Add edges
        for edge in edges:
            G.add_edge(
                edge['source'],
                edge['target'],
                weight=edge.get('weight', 1.0)
            )
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Get position layout
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes by domain
        domains = {}
        for node_id, attrs in G.nodes(data=True):
            domain = attrs.get('domain', 'unknown')
            if domain not in domains:
                domains[domain] = []
            domains[domain].append(node_id)
        
        for domain, node_ids in domains.items():
            color = self.node_colors.get(domain, self.default_color)
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=node_ids,
                node_color=color,
                node_size=[20 + G.degree(n) * 5 for n in node_ids],
                alpha=0.8,
                label=domain
            )
        
        # Draw edges with width based on weight
        nx.draw_networkx_edges(
            G, pos,
            width=[d['weight'] * 2 for u, v, d in G.edges(data=True)],
            alpha=0.4,
            edge_color='gray'
        )
        
        # Draw node labels
        nx.draw_networkx_labels(
            G, pos,
            labels={n: n.split('_')[1] if '_' in n else n for n in G.nodes()},
            font_size=8,
            font_color='black'
        )
        
        # Set title and legend
        plt.title('Kaleidoscope AI Network Structure')
        plt.legend(title="Domains")
        plt.axis('off')
        plt.tight_layout()
        
        # Save to file if specified
        if output_file:
            output_path = os.path.join(self.output_dir, output_file)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            return output_path
        
        # Otherwise return as base64 encoded PNG
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        buffer.seek(0)
        encoded = base64.b64encode(buffer.read()).decode()
        return encoded
    
    def visualize_insights(self, 
                         insights_data: Dict,
                         output_file: str = None,
                         interactive: bool = True) -> Union[str, None]:
        """
        Visualize insights generated by the Kaleidoscope system.
        
        Args:
            insights_data: Insights data from the API
            output_file: Path to save the visualization (optional)
            interactive: Whether to create an interactive visualization
        
        Returns:
            Path to output file if saved, None otherwise
        """
        if interactive:
            return self._plotly_insights_visualization(insights_data, output_file)
        else:
            return self._matplotlib_insights_visualization(insights_data, output_file)
    
    def _plotly_insights_visualization(self, 
                                    insights_data: Dict,
                                    output_file: str = None) -> str:
        """Create an interactive insights visualization using Plotly."""
        # Extract insights from data
        all_insights = []
        
        # Process different types of insight structures
        if 'insights' in insights_data:
            all_insights.extend(insights_data['insights'])
        elif 'patterns' in insights_data:
            all_insights.extend(insights_data['patterns'])
        elif 'outputs' in insights_data:
            for output in insights_data['outputs']:
                if 'result' in output and 'insights' in output['result']:
                    all_insights.extend(output['result']['insights'])
        
        if not all_insights:
            # No insights found
            return None
        
        # Convert insights to dataframe for easier handling
        insights_list = []
        
        for insight in all_insights:
            insight_type = insight.get('type', 'unknown')
            confidence = insight.get('confidence', 0.5)
            description = insight.get('description', 'No description')
            
            # Extract additional relevant fields
            source = insight.get('node_id', insight.get('source_gear', 'unknown'))
            timestamp = insight.get('timestamp', 0)
            
            insights_list.append({
                'type': insight_type,
                'confidence': confidence,
                'description': description,
                'source': source,
                'timestamp': timestamp
            })
        
        df = pd.DataFrame(insights_list)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [{"type": "bar"}, {"type": "pie"}],
                [{"type": "scatter", "colspan": 2}, None]
            ],
            subplot_titles=(
                "Confidence by Insight Type",
                "Insight Type Distribution",
                "Insights Timeline"
            ),
            vertical_spacing=0.1,
            horizontal_spacing=0.05
        )
        
        # 1. Bar chart: Confidence by Insight Type
        if len(df) > 0 and 'type' in df.columns and 'confidence' in df.columns:
            insight_type_confidence = df.groupby('type')['confidence'].mean().reset_index()
            
            fig.add_trace(
                go.Bar(
                    x=insight_type_confidence['type'],
                    y=insight_type_confidence['confidence'],
                    marker_color='rgb(55, 83, 109)',
                    text=insight_type_confidence['confidence'].round(2),
                    textposition='auto'
                ),
                row=1, col=1
            )
            
            fig.update_xaxes(title_text="Insight Type", row=1, col=1)
            fig.update_yaxes(title_text="Avg. Confidence", range=[0, 1], row=1, col=1)
        
        # 2. Pie chart: Insight Type Distribution
        if len(df) > 0 and 'type' in df.columns:
            type_counts = df['type'].value_counts()
            
            fig.add_trace(
                go.Pie(
                    labels=type_counts.index,
                    values=type_counts.values,
                    hole=0.3
                ),
                row=1, col=2
            )
        
        # 3. Scatter plot: Insights Timeline
        if len(df) > 0 and 'timestamp' in df.columns and 'confidence' in df.columns:
            # Sort by timestamp
            df_sorted = df.sort_values('timestamp')
            
            fig.add_trace(
                go.Scatter(
                    x=df_sorted['timestamp'],
                    y=df_sorted['confidence'],
                    mode='markers+lines',
                    marker=dict(
                        size=10,
                        color=df_sorted['confidence'],
                        colorscale=self.colorscale,
                        showscale=True,
                        colorbar=dict(title="Confidence")
                    ),
                    text=df_sorted['description'],
                    hoverinfo="text+y"
                ),
                row=2, col=1
            )
            
            fig.update_xaxes(title_text="Timestamp", row=2, col=1)
            fig.update_yaxes(title_text="Confidence", range=[0, 1], row=2, col=1)
        
        # Set layout
        fig.update_layout(
            title_text="Kaleidoscope AI Insights Analysis",
            height=800,
            showlegend=False
        )
        
        # Save to file if specified
        if output_file:
            output_path = os.path.join(self.output_dir, output_file)
            fig.write_html(output_path)
            return output_path
        
        # Otherwise return as base64 encoded HTML
        buffer = io.StringIO()
        fig.write_html(buffer)
        html_str = buffer.getvalue()
        
        encoded = base64.b64encode(html_str.encode()).decode()
        return encoded
    
    def _matplotlib_insights_visualization(self, 
                                        insights_data: Dict,
                                        output_file: str = None) -> str:
        """Create a static insights visualization using Matplotlib."""
        # Extract insights from data
        all_insights = []
        
        # Process different types of insight structures
        if 'insights' in insights_data:
            all_insights.extend(insights_data['insights'])
        elif 'patterns' in insights_data:
            all_insights.extend(insights_data['patterns'])
        elif 'outputs' in insights_data:
            for output in insights_data['outputs']:
                if 'result' in output and 'insights' in output['result']:
                    all_insights.extend(output['result']['insights'])
        
        if not all_insights:
            # No insights found
            return None
        
        # Convert insights to dataframe for easier handling
        insights_list = []
        
        for insight in all_insights:
            insight_type = insight.get('type', 'unknown')
            confidence = insight.get('confidence', 0.5)
            description = insight.get('description', 'No description')
            
            # Extract additional relevant fields
            source = insight.get('node_id', insight.get('source_gear', 'unknown'))
            timestamp = insight.get('timestamp', 0)
            
            insights_list.append({
                'type': insight_type,
                'confidence': confidence,
                'description': description,
                'source': source,
                'timestamp': timestamp
            })
        
        df = pd.DataFrame(insights_list)
        
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Bar chart: Confidence by Insight Type
        if len(df) > 0 and 'type' in df.columns and 'confidence' in df.columns:
            insight_type_confidence = df.groupby('type')['confidence'].mean()
            
            bars = insight_type_confidence.plot(
                kind='bar',
                ax=axs[0, 0],
                color='skyblue',
                edgecolor='navy',
                alpha=0.7
            )
            
            # Add value labels
            for bar in bars.patches:
                height = bar.get_height()
                axs[0, 0].text(
                    bar.get_x() + bar.get_width() / 2.,
                    height + 0.01,
                    f'{height:.2f}',
                    ha='center', va='bottom',
                    fontsize=9
                )
            
            axs[0, 0].set_title('Average Confidence by Insight Type')
            axs[0, 0].set_xlabel('Insight Type')
            axs[0, 0].set_ylabel('Confidence')
            axs[0, 0].set_ylim(0, 1)
            axs[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Pie chart: Insight Type Distribution
        if len(df) > 0 and 'type' in df.columns:
            type_counts = df['type'].value_counts()
            
            axs[0, 1].pie(
                type_counts.values,
                labels=type_counts.index,
                autopct='%1.1f%%',
                startangle=90,
                shadow=True,
                wedgeprops={'alpha': 0.7}
            )
            axs[0, 1].set_title('Insight Type Distribution')
            axs[0, 1].axis('equal')
        
        # 3. Scatter plot: Insights Timeline
        if len(df) > 0 and 'timestamp' in df.columns and 'confidence' in df.columns:
            # Sort by timestamp
            df_sorted = df.sort_values('timestamp')
            
            # Create colormap
            cmap = plt.cm.get_cmap('viridis')
            
            # Scatter plot
            scatter = axs[1, 0].scatter(
                df_sorted['timestamp'],
                df_sorted['confidence'],
                c=df_sorted['confidence'],
                cmap=cmap,
                alpha=0.7,
                s=100,
                edgecolor='k'
            )
            
            # Add colorbar
            plt.colorbar(scatter, ax=axs[1, 0], label='Confidence')
            
            # Add trend line
            axs[1, 0].plot(
                df_sorted['timestamp'],
                df_sorted['confidence'],
                'r--',
                alpha=0.3
            )
            
            axs[1, 0].set_title('Insights Timeline')
            axs[1, 0].set_xlabel('Timestamp')
            axs[1, 0].set_ylabel('Confidence')
            axs[1, 0].set_ylim(0, 1)
        
        # 4. Table: Top Insights
        if len(df) > 0:
            # Sort by confidence and get top 5
            top_insights = df.sort_values('confidence', ascending=False).head(5)
            
            # Clear the axis for the table
            axs[1, 1].axis('off')
            
            # Create table data
            table_data = []
            for _, row in top_insights.iterrows():
                table_data.append([
                    row['type'],
                    f"{row['confidence']:.2f}",
                    row['description'][:30] + '...' if len(row['description']) > 30 else row['description']
                ])
            
            # Create table
            table = axs[1, 1].table(
                cellText=table_data,
                colLabels=['Type', 'Conf.', 'Description'],
                loc='center',
                cellLoc='center',
                colWidths=[0.2, 0.1, 0.7]
            )
            
            # Style table
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)
            
            axs[1, 1].set_title('Top Insights by Confidence')
        
        # Set overall title
        fig.suptitle('Kaleidoscope AI Insights Analysis', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save to file if specified
        if output_file:
            output_path = os.path.join(self.output_dir, output_file)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            return output_path
        
        # Otherwise return as base64 encoded PNG
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        buffer.seek(0)
        encoded = base64.b64encode(buffer.read()).decode()
        return encoded
    
    def visualize_molecular_structure(self, 
                                    molecule_data: Dict,
                                    output_file: str = None) -> str:
        """
        Visualize molecular structure and properties.
        
        Args:
            molecule_data: Molecule data from the API
            output_file: Path to save the visualization (optional)
        
        Returns:
            Path to output file if saved, base64 encoded HTML otherwise
        """
        # Extract molecule information
        smiles = molecule_data.get('smiles')
        if not smiles:
            return None
        
        # Create molecule from SMILES
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            from rdkit.Chem import Draw
            from rdkit.Chem.Draw import rdMolDraw2D
            
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return None
            
            # Generate 2D coordinates
            AllChem.Compute2DCoords(mol)
            
            # Create figure with subplots
            fig = make_subplots(
                rows=2, cols=2,
                specs=[
                    [{"type": "image"}, {"type": "table"}],
                    [{"type": "bar", "colspan": 2}, None]
                ],
                subplot_titles=(
                    "Molecular Structure",
                    "Properties",
                    "Property Values"
                ),
                vertical_spacing=0.1,
                horizontal_spacing=0.05
            )
            
            # 1. Molecular structure image
            drawer = rdMolDraw2D.MolDraw2DCairo(400, 400)
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            png_data = drawer.GetDrawingText()
            
            # Encode PNG image for Plotly
            encoded_image = base64.b64encode(png_data).decode('ascii')
            
            fig.add_trace(
                go.Image(
                    source=f'data:image/png;base64,{encoded_image}'
                ),
                row=1, col=1
            )
            
            # 2. Properties table
            properties = molecule_data.get('properties', {})
            if properties:
                property_names = list(properties.keys())
                property_values = [properties[p] for p in property_names]
                
                fig.add_trace(
                    go.Table(
                        header=dict(
                            values=['Property', 'Value'],
                            fill_color='paleturquoise',
                            align='left',
                            font=dict(size=12)
                        ),
                        cells=dict(
                            values=[property_names, 
                                   [f"{v:.2f}" if isinstance(v, float) else v for v in property_values]],
                            fill_color='lavender',
                            align='left',
                            font=dict(size=11)
                        )
                    ),
                    row=1, col=2
                )
            
            # 3. Property values bar chart
            if properties:
                # Filter numerical properties
                num_properties = {k: v for k, v in properties.items() 
                               if isinstance(v, (int, float))}
                
                if num_properties:
                    prop_names = list(num_properties.keys())
                    prop_values = list(num_properties.values())
                    
                    fig.add_trace(
                        go.Bar(
                            x=prop_names,
                            y=prop_values,
                            marker_color='rgb(55, 83, 109)',
                            text=[f"{v:.2f}" if isinstance(v, float) else v for v in prop_values],
                            textposition='auto'
                        ),
                        row=2, col=1
                    )
                    
                    fig.update_xaxes(title_text="Property", row=2, col=1)
                    fig.update_yaxes(title_text="Value", row=2, col=1)
            
            # Set layout
            fig.update_layout(
                title_text=f"Molecule: {molecule_data.get('formula', smiles)}",
                height=800,
                showlegend=False
            )
            
            # Save to file if specified
            if output_file:
                output_path = os.path.join(self.output_dir, output_file)
                fig.write_html(output_path)
                return output_path
            
            # Otherwise return as base64 encoded HTML
            buffer = io.StringIO()
            fig.write_html(buffer)
            html_str = buffer.getvalue()
            
            encoded = base64.b64encode(html_str.encode()).decode()
            return encoded
            
        except ImportError:
            # RDKit not available, create a simple figure instead
            fig = go.Figure()
            
            # Add SMILES as text
            fig.add_annotation(
                text=f"SMILES: {smiles}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            
            # Add properties
            properties = molecule_data.get('properties', {})
            if properties:
                props_text = "<br>".join([f"{k}: {v}" for k, v in properties.items()])
                
                fig.add_annotation(
                    text=props_text,
                    xref="paper", yref="paper",
                    x=0.5, y=0.3,
                    showarrow=False,
                    font=dict(size=14)
                )
            
            # Set layout
            fig.update_layout(
                title_text=f"Molecule: {molecule_data.get('formula', smiles)}",
                height=600
            )
            
            # Save to file if specified
            if output_file:
                output_path = os.path.join(self.output_dir, output_file)
                fig.write_html(output_path)
                return output_path
            
            # Otherwise return as base64 encoded HTML
            buffer = io.StringIO()
            fig.write_html(buffer)
            html_str = buffer.getvalue()
            
            encoded = base64.b64encode(html_str.encode()).decode()
            return encoded

    def visualize_docking_results(self, 
                               docking_data: Dict,
                               output_file: str = None) -> str:
        """
        Visualize molecular docking results.
        
        Args:
            docking_data: Docking results data from the API
            output_file: Path to save the visualization (optional)
        
        Returns:
            Path to output file if saved, base64 encoded HTML otherwise
        """
        # Extract docking information
        molecule_id = docking_data.get('molecule_id')
        target = docking_data.get('target')
        docking_score = docking_data.get('docking_score')
        binding_modes = docking_data.get('binding_modes', [])
        
        if not binding_modes:
            return None
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [{"type": "table"}, {"type": "bar"}],
                [{"type": "scatter", "colspan": 2}, None]
            ],
            subplot_titles=(
                "Binding Modes",
                "Docking Scores",
                "Structure-Activity Relationship"
            ),
            vertical_spacing=0.1,
            horizontal_spacing=0.05
        )
        
        # 1. Binding modes table
        mode_ids = [mode.get('mode_id', f"mode_{i}") for i, mode in enumerate(binding_modes)]
        scores = [mode.get('score', 0) for mode in binding_modes]
        rmsds = [mode.get('rmsd', 0) for mode in binding_modes]
        h_bonds = [mode.get('h_bonds', 0) for mode in binding_modes]
        interactions = [mode.get('interactions', '') for mode in binding_modes]
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['Mode', 'Score', 'RMSD', 'H-Bonds', 'Interactions'],
                    fill_color='paleturquoise',
                    align='left',
                    font=dict(size=12)
                ),
                cells=dict(
                    values=[
                        [id.split('_')[-1] if '_' in id else id for id in mode_ids],
                        [f"{s:.2f}" for s in scores],
                        [f"{r:.2f}" for r in rmsds],
                        h_bonds,
                        interactions
                    ],
                    fill_color='lavender',
                    align='left',
                    font=dict(size=11)
                )
            ),
            row=1, col=1
        )
        
        # 2. Docking scores bar chart
        fig.add_trace(
            go.Bar(
                x=[f"Mode {i+1}" for i in range(len(binding_modes))],
                y=scores,
                marker_color=['blue' if i == 0 else 'lightblue' for i in range(len(binding_modes))],
                text=[f"{s:.2f}" for s in scores],
                textposition='auto'
            ),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Binding Mode", row=1, col=2)
        fig.update_yaxes(title_text="Docking Score", row=1, col=2)
        
        # 3. Structure-Activity Relationship scatter plot
        fig.add_trace(
            go.Scatter(
                x=rmsds,
                y=scores,
                mode='markers+text',
                marker=dict(
                    size=[10 + h*3 for h in h_bonds],
                    color=scores,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Score")
                ),
                text=[f"Mode {i+1}" for i in range(len(binding_modes))],
                textposition='top center',
                hovertext=[f"Mode {i+1}<br>Score: {s:.2f}<br>RMSD: {r:.2f}<br>H-Bonds: {h}" 
                         for i, (s, r, h) in enumerate(zip(scores, rmsds, h_bonds))],
                hoverinfo='text'
            ),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="RMSD (Å)", row=2, col=1)
        fig.update_yaxes(title_text="Docking Score", row=2, col=1)
        
        # Set layout
        fig.update_layout(
            title_text=f"Docking Results: {molecule_id} with {target}",
            height=800,
            showlegend=False
        )
        
        # Add annotation for overall docking score
        fig.add_annotation(
            text=f"Overall Docking Score: {docking_score:.2f}",
            xref="paper", yref="paper",
            x=0.5, y=1.05,
            showarrow=False,
            font=dict(size=14)
        )
        
        # Save to file if specified
        if output_file:
            output_path = os.path.join(self.output_dir, output_file)
            fig.write_html(output_path)
            return output_path
        
        # Otherwise return as base64 encoded HTML
        buffer = io.StringIO()
        fig.write_html(buffer)
        html_str = buffer.getvalue()
        
        encoded = base64.b64encode(html_str.encode()).decode()
        return encoded
    
    def visualize_pattern_analysis(self, 
                                 pattern_data: Dict,
                                 output_file: str = None) -> str:
        """
        Visualize pattern analysis results.
        
        Args:
            pattern_data: Pattern analysis data from the API
            output_file: Path to save the visualization (optional)
        
        Returns:
            Path to output file if saved, base64 encoded HTML otherwise
        """
        # Extract pattern information
        patterns = pattern_data.get('patterns', [])
        relationships = pattern_data.get('relationships', [])
        
        if not patterns:
            return None
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [{"type": "bar"}, {"type": "pie"}],
                [{"type": "network", "colspan": 2}, None]
            ],
            subplot_titles=(
                "Pattern Confidence by Type",
                "Pattern Type Distribution",
                "Pattern Relationships Network"
            ),
            vertical_spacing=0.1,
            horizontal_spacing=0.05
        )
        
        # 1. Pattern confidence by type
        pattern_types = [p.get('type', 'unknown') for p in patterns]
        confidences = [p.get('confidence', 0.5) for p in patterns]
        
        # Aggregate confidence by type
        df = pd.DataFrame({'type': pattern_types, 'confidence': confidences})
        type_confidence = df.groupby('type')['confidence'].mean().reset_index()
        
        fig.add_trace(
            go.Bar(
                x=type_confidence['type'],
                y=type_confidence['confidence'],
                marker_color='rgb(55, 83, 109)',
                text=type_confidence['confidence'].round(2),
                textposition='auto'
            ),
            row=1, col=1
        )
        
        fig.update_xaxes(title_text="Pattern Type", row=1, col=1)
        fig.update_yaxes(title_text="Avg. Confidence", range=[0, 1], row=1, col=1)
        
        # 2. Pattern type distribution pie chart
        type_counts = df['type'].value_counts()
        
        fig.add_trace(
            go.Pie(
                labels=type_counts.index,
                values=type_counts.values,
                hole=0.3
            ),
            row=1, col=2
        )
        
        # 3. Pattern relationships network
        if relationships:
            # Create networkx graph
            G = nx.Graph()
            
            # Add nodes (patterns)
            for pattern in patterns:
                pattern_id = pattern.get('pattern_id')
                pattern_type = pattern.get('type', 'unknown')
                confidence = pattern.get('confidence', 0.5)
                
                if pattern_id:
                    G.add_node(
                        pattern_id,
                        type=pattern_type,
                        confidence=confidence
                    )
            
            # Add edges (relationships)
            for rel in relationships:
                source = rel.get('pattern1_id')
                target = rel.get('pattern2_id')
                rel_type = rel.get('relationship_type', 'unknown')
                strength = rel.get('strength', 0.5)
                
                if source and target:
                    G.add_edge(
                        source,
                        target,
                        type=rel_type,
                        weight=strength
                    )
            
            # Create network visualization
            if len(G.nodes) > 0:
                # Get position layout
                pos = nx.spring_layout(G, seed=42)
                
                # Prepare node data
                node_x, node_y = [], []
                node_text, node_colors, node_sizes = [], [], []
                
                for node, attrs in G.nodes(data=True):
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    
                    # Node text
                    node_text.append(
                        f"ID: {node}<br>"
                        f"Type: {attrs.get('type', 'unknown')}<br>"
                        f"Confidence: {attrs.get('confidence', 0.5):.2f}"
                    )
                    
                    # Node color by type
                    pattern_type = attrs.get('type', 'unknown')
                    color_hash = hash(pattern_type) % 100000
                    node_colors.append(f"rgb({color_hash % 256}, {(color_hash // 256) % 256}, {(color_hash // 65536) % 256})")
                    
                    # Node size by confidence
                    confidence = attrs.get('confidence', 0.5)
                    node_sizes.append(10 + confidence * 20)
                
                # Add nodes to plot
                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers',
                    marker=dict(
                        size=node_sizes,
                        color=node_colors,
                        line=dict(width=1, color='#ffffff')
                    ),
                    text=node_text,
                    hoverinfo='text'
                )
                
                # Prepare edge data
                edge_x, edge_y = [], []
                edge_colors, edge_widths = [], []
                
                for edge in G.edges(data=True):
                    source, target = edge[0], edge[1]
                    x0, y0 = pos[source]
                    x1, y1 = pos[target]
                    
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                    
                    # Edge width by strength
                    weight = edge[2].get('weight', 0.5)
                    edge_widths.extend([weight * 5, weight * 5, 0])
                    
                    # Edge color by type
                    rel_type = edge[2].get('type', 'unknown')
                    color_hash = hash(rel_type) % 100000
                    edge_color = f"rgba({color_hash % 256}, {(color_hash // 256) % 256}, {(color_hash // 65536) % 256}, 0.7)"
                    edge_colors.extend([color_hash / 100000, color_hash / 100000, 0])
                
                # Add edges to plot
                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    mode='lines',
                    line=dict(
                        width=edge_widths,
                        color=edge_colors,
                        colorscale=self.colorscale
                    ),
                    hoverinfo='none'
                )
                
                # Add traces to figure
                fig.add_trace(edge_trace, row=2, col=1)
                fig.add_trace(node_trace, row=2, col=1)
                
                # Update axes
                fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=2, col=1)
                fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, row=2, col=1)
        
        # Set layout
        fig.update_layout(
            title_text="Pattern Analysis Results",
            height=800,
            showlegend=False
        )
        
        # Save to file if specified
        if output_file:
            output_path = os.path.join(self.output_dir, output_file)
            fig.write_html(output_path)
            return output_path
        
        # Otherwise return as base64 encoded HTML
        buffer = io.StringIO()
        fig.write_html(buffer)
        html_str = buffer.getvalue()
        
        encoded = base64.b64encode(html_str.encode()).decode()
        return encoded


# Visualization helper function
def create_visualizer(output_dir: str = './visualizations') -> KaleidoscopeVisualizer:
    """Create a Kaleidoscope visualizer instance."""
    return KaleidoscopeVisualizer(output_dir=output_dir)
