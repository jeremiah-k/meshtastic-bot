# Note: Development of this project has been moved to a self-hosted GitLab instance. You can start a discussion here if you want to collaborate on this project.

# Meshtastic Chatbot

This repository contains the Meshtastic Chatbot project which utilizes the Meshtastic documentation as a submodule.

## Installation

Clone the repository and its submodules.

1. Open your terminal.
2. Run the following command to clone the repository and its submodules:

```bash
git clone --recurse-submodules https://github.com/jeremiah-k/meshtastic-bot.git
```

## Setup

Create a Python virtual environment in the project directory.

```
python3 -m venv .pyenv
```

Activate the virtual environment and install the project dependencies.

```
source .pyenv/bin/activate
pip install -r requirements.txt
```