#!/bin/bash

# Start the CRM MCP server in the background
python mcp_server/crm_server.py &

# Start the Streamlit app on port 8080
streamlit run app.py --server.port 8080 --server.address 0.0.0.0
