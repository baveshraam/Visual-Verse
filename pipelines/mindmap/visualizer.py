"""
Mind Map Visualizer - PREMIUM VERSION
Professional, clean hierarchical mind map visualization.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
import io

try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

from .graph_builder import GraphBuilder, NodeType, ConceptEdge
from .relation_extractor import RelationType


# Premium color schemes
THEMES = {
    "light": {
        "bgcolor": "#f8f9fa",
        "central": "#dc3545",      # Red
        "primary": "#0d6efd",      # Blue  
        "secondary": "#198754",    # Green
        "tertiary": "#6c757d",     # Gray
        "edge_default": "#adb5bd",
        "edge_is_a": "#0d6efd",    # Blue for hierarchy
        "edge_uses": "#fd7e14",    # Orange for uses
        "font": "#212529"
    },
    "dark": {
        "bgcolor": "#1a1a2e",
        "central": "#e74c3c",
        "primary": "#3498db",
        "secondary": "#27ae60",
        "tertiary": "#7f8c8d",
        "edge_default": "#4a5568",
        "edge_is_a": "#3498db",
        "edge_uses": "#f39c12",
        "font": "#ecf0f1"
    }
}


class MindMapVisualizer:
    """
    Creates professional hierarchical mind maps.
    
    Features:
    - Clean tree layout (no physics jitter)
    - Color-coded relationships
    - Proper edge labels
    - Shadow effects for depth
    """
    
    def __init__(self, theme: str = "light"):
        if not PYVIS_AVAILABLE:
            raise ImportError("PyVis required")
        if not NETWORKX_AVAILABLE:
            raise ImportError("NetworkX required")
        
        self.theme = THEMES.get(theme, THEMES["light"])
        self.network = None
        self.graph_builder = None
    
    def set_theme(self, theme: str):
        """Set visualization theme."""
        self.theme = THEMES.get(theme, THEMES["light"])
    
    def _get_node_style(self, node_type: str) -> Tuple[str, int, str]:
        """Get color, size, and shape for node type."""
        styles = {
            "central": (self.theme["central"], 55, "dot"),
            "primary": (self.theme["primary"], 40, "dot"),
            "secondary": (self.theme["secondary"], 30, "dot"),
            "tertiary": (self.theme["tertiary"], 22, "dot")
        }
        return styles.get(node_type, styles["tertiary"])
    
    def _get_edge_color(self, relation_type: str) -> str:
        """Get edge color based on relation type."""
        if relation_type in ["is_a", "part_of"]:
            return self.theme["edge_is_a"]
        elif relation_type in ["uses"]:
            return self.theme["edge_uses"]
        else:
            return self.theme["edge_default"]
    
    def _create_network(self) -> Network:
        """Create PyVis network with professional settings."""
        net = Network(
            width="100%",
            height="600px",
            bgcolor=self.theme["bgcolor"],
            font_color=self.theme["font"],
            directed=True,
            notebook=False
        )
        
        # Professional hierarchical layout
        options = """
        {
            "nodes": {
                "font": {
                    "size": 16,
                    "face": "Inter, Arial, sans-serif",
                    "color": "%s",
                    "strokeWidth": 2,
                    "strokeColor": "white"
                },
                "borderWidth": 3,
                "borderWidthSelected": 4,
                "shadow": {
                    "enabled": true,
                    "color": "rgba(0,0,0,0.3)",
                    "size": 8,
                    "x": 3,
                    "y": 3
                }
            },
            "edges": {
                "arrows": {
                    "to": {
                        "enabled": true,
                        "scaleFactor": 0.6
                    }
                },
                "smooth": {
                    "type": "cubicBezier",
                    "forceDirection": "vertical",
                    "roundness": 0.5
                },
                "width": 2,
                "font": {
                    "size": 11,
                    "align": "middle",
                    "background": "white"
                }
            },
            "physics": {
                "enabled": false
            },
            "layout": {
                "hierarchical": {
                    "enabled": true,
                    "direction": "UD",
                    "sortMethod": "directed",
                    "nodeSpacing": 200,
                    "levelSeparation": 150,
                    "treeSpacing": 250,
                    "blockShifting": true,
                    "edgeMinimization": true
                }
            },
            "interaction": {
                "hover": true,
                "tooltipDelay": 100,
                "zoomView": true,
                "dragView": true
            }
        }
        """ % self.theme["font"]
        
        net.set_options(options)
        return net
    
    def visualize(self, graph_builder: GraphBuilder, title: str = "Mind Map") -> Network:
        """Create visualization from GraphBuilder."""
        self.graph_builder = graph_builder
        self.network = self._create_network()
        
        # Add nodes with styling
        for node_id, node in graph_builder.nodes.items():
            color, size, shape = self._get_node_style(node.node_type.value)
            
            # Tooltip with details
            tooltip = f"<div style='font-family: Arial; padding: 8px;'>"
            tooltip += f"<b style='font-size: 14px;'>{node.label}</b><br>"
            tooltip += f"<span style='color: #666;'>Type: {node.node_type.value}</span>"
            tooltip += "</div>"
            
            self.network.add_node(
                node_id,
                label=node.label,
                color={
                    "background": color,
                    "border": color,
                    "highlight": {"background": color, "border": "#000"}
                },
                size=size,
                shape=shape,
                title=tooltip,
                font={
                    "size": 16 if node.node_type == NodeType.CENTRAL else 14,
                    "color": self.theme["font"],
                    "strokeWidth": 2,
                    "strokeColor": "#fff"
                }
            )
        
        # Add edges with relationship colors
        for edge in graph_builder.edges:
            edge_color = self._get_edge_color(edge.relation_type.value)
            
            # Show label for important relationships
            label = ""
            if edge.relation_type in [RelationType.USES]:
                label = "uses"
            
            self.network.add_edge(
                edge.source_id,
                edge.target_id,
                color=edge_color,
                width=2.5 if edge.relation_type == RelationType.IS_A else 2,
                title=edge.relation_type.value.replace("_", " "),
                label=label,
                arrows={"to": {"enabled": True, "scaleFactor": 0.6}}
            )
        
        return self.network
    
    def get_html_string(self) -> Optional[str]:
        """Get HTML for embedding."""
        if not self.network:
            return None
        try:
            return self.network.generate_html()
        except:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
                self.network.save_graph(f.name)
                with open(f.name, 'r', encoding='utf-8') as hf:
                    return hf.read()
    
    def save_html(self, filepath: Path) -> bool:
        """Save to HTML file."""
        if not self.network:
            return False
        try:
            self.network.save_graph(str(filepath))
            return True
        except:
            return False
    
    def generate_png(self, graph_builder: GraphBuilder) -> bytes:
        """Generate PNG using matplotlib with proper hierarchy."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            
            fig, ax = plt.subplots(figsize=(14, 10), facecolor='white')
            ax.set_facecolor('white')
            
            graph = graph_builder.graph
            
            if len(graph.nodes()) == 0:
                ax.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=24)
                ax.axis('off')
            else:
                # Use hierarchical layout
                try:
                    # Try graphviz-style layout for proper hierarchy
                    pos = nx.spring_layout(graph, k=3, iterations=100, seed=42)
                except:
                    pos = nx.shell_layout(graph)
                
                # Prepare node properties
                node_colors = []
                node_sizes = []
                labels = {}
                
                for node_id in graph.nodes():
                    node = graph_builder.nodes.get(node_id)
                    if node:
                        color, size, _ = self._get_node_style(node.node_type.value)
                        node_colors.append(color)
                        node_sizes.append(size * 50)
                        labels[node_id] = node.label
                    else:
                        node_colors.append(self.theme["tertiary"])
                        node_sizes.append(1000)
                        labels[node_id] = node_id
                
                # Draw edges with colors based on type
                for edge in graph_builder.edges:
                    edge_color = self._get_edge_color(edge.relation_type.value)
                    if graph.has_edge(edge.source_id, edge.target_id):
                        nx.draw_networkx_edges(
                            graph, pos, ax=ax,
                            edgelist=[(edge.source_id, edge.target_id)],
                            edge_color=edge_color,
                            arrows=True,
                            arrowsize=20,
                            width=2.5,
                            alpha=0.8,
                            connectionstyle="arc3,rad=0.1"
                        )
                
                # Draw nodes
                nx.draw_networkx_nodes(
                    graph, pos, ax=ax,
                    node_color=node_colors,
                    node_size=node_sizes,
                    alpha=0.95,
                    edgecolors='white',
                    linewidths=3
                )
                
                # Draw labels
                nx.draw_networkx_labels(
                    graph, pos, labels, ax=ax,
                    font_size=11,
                    font_weight='bold',
                    font_color='#1a1a2e'
                )
                
                ax.axis('off')
                
                # Legend
                legend_elements = [
                    mpatches.Patch(color=self.theme["central"], label='Central Topic'),
                    mpatches.Patch(color=self.theme["primary"], label='Main Concept'),
                    mpatches.Patch(color=self.theme["secondary"], label='Sub-Concept'),
                    mpatches.Patch(color=self.theme["tertiary"], label='Detail'),
                ]
                ax.legend(handles=legend_elements, loc='upper left', 
                         fontsize=10, framealpha=0.95, edgecolor='none')
            
            plt.tight_layout()
            
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close(fig)
            
            buffer.seek(0)
            return buffer.getvalue()
            
        except Exception as e:
            print(f"PNG error: {e}")
            import traceback
            traceback.print_exc()
            return b""


# Test
if __name__ == "__main__":
    print("MindMapVisualizer module loaded")
