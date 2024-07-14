
# Project Setup

This project requires installing dependencies listed in a `requirements.txt` file and then running a `client.py` script. The setup process has been automated with a shell script.

## Prerequisites

- Ensure you have Python and pip installed on your system. You can verify the installations by running:
  ```
  python3 --version
  pip --version`` 
## Setup Instructions
    
1.  **Make the Script Executable**
    
    Open a terminal, navigate to the directory where the extracted files are located, and run the following command to make the script executable:
    
       
    `chmod +x install_and_run.sh` 
    
3.  **Run the Script**
    
    Execute the script to install the dependencies and run `client.py`:
   
    `./install_and_run.sh` 
    

## Troubleshooting

-   Ensure that `requirements.txt` and `client.py` are present in the same directory as the `install_and_run.sh` script.
-   Verify that Python and pip are correctly installed and available in your system's PATH.

If you encounter any issues, please check the error messages and ensure all prerequisites are met.