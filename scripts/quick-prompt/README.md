# Quick prompt script

## Features

- Quick offline terminal chat with open source LMs
- Optionally save conversations to file

## Supported model file formats

- torch and safetensors models

## Usage

Configure your chat using the `.env` file. The below table outlines the configurations:

| Environment Variable | Usage                                                  |
| -------------------- | ------------------------------------------------------ |
| HF_MODEL             | The Hugging Face model you want to start a chat with   |
| SAVE_CONVERSATIONS   | Set to "True" to save conversations to your local host |

Run the `uv` executable script. For example:

```bash
cd scripts/quick-prompt-container
source .env
./chat.py
```

The container chat terminal will start an your prompts will be processed by the model and conversations
saved (if `SAVE_CONVERSATIONS="True"`) to the `conversations/` directory.
