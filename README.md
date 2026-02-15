# Modular Steering Vectors 


## Installation

0. Assuming you have already cloned this repository, clone the `llm-steer-instruct` repository somewhere on your machine
```bash
   cd /SOME/PATH/ON/YOUR/MACHINE
   git clone https://github.com/microsoft/llm-steer-instruct.git 
```

1. Go back to the home folder of this project and create a `.env` file with the following content:
```bash
    INSTRUCT_REPO_PATH=/SOME/PATH/ON/YOUR/MACHINE/llm-steer-instruct
    TOGETHER_API_KEY=YOUR_TOGETHER_API_KEY
    LLAMA_GUARD2_LOCAL=0 # [0,1] if local then you need at least 40GB of VRAM
```