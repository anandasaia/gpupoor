#!/bin/bash

# Install the contents of requirements.txt
pip install -r requirements.txt

# Check if client.py exists in the same directory as requirements.txt
if [ -f "client.py" ]; then
  # If client.py exists, run it
  python client.py
else
  # If client.py does not exist, print a message
  echo "client.py not found in the current directory."
fi
