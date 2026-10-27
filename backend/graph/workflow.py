# langgraph imports
from langgraph.graph import StateGraph,START, END
# from langgraph.checkpoint.sqlite import SqliteSaver

from backend.graph.state import AgentState
from backend.graph.nodes import planner_node, model_training_node, check_status, evaluation_node, export_node

# Build StateGraph
graph = StateGraph(AgentState)

# Add Nodes
graph.add_node("planner_node", planner_node)
graph.add_node("model_training_node", model_training_node)
graph.add_node("evaluation_node", evaluation_node)
graph.add_node("export_node", export_node)

# Entrypoint and Edges
graph.add_edge(START, "planner_node")
graph.add_edge("planner_node", "model_training_node")
graph.add_conditional_edges("model_training_node", check_status)

graph.add_edge("evaluation_node", "export_node")
graph.add_edge("export_node", END)

# Compile Graph
app = graph.compile()