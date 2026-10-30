## 🚀 How to Use Anlytx-AI with Claude Desktop

Because Claude Desktop features an internal code sandbox, it may occasionally try to analyze datasets locally unless directed otherwise. To ensure Claude routes all requests through the remote **Anlytx-AI** MCP pipeline, use the recommended prompt structure below.

### Recommended System / User Prompt:
> *"Using the connected anlytx-ai MCP tools, process the attached dataset. Do not use local Python code execution; instead, call the remote tools step-by-step starting with inspect_dataset_summary."*


# git add . ; git commit -m "update" ; git push