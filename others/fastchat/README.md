# Requirements

```
pip install "fschat[model_worker,webui]"
```

# Deploying LLM
```
# The first step is to start the controller service： 
python -m fastchat.serve.controller

# worker 0
python -m fastchat.serve.model_worker --model-name {model1} --model-path {model1_path} --controller http://localhost:21001 --port 31000 --worker http://localhost:31000
# worker 1
python -m fastchat.serve.model_worker --model-name {model2} --model-path {model2_path} --controller http://localhost:21001 --port 31001 --worker http://localhost:31001

# Verification：
python -m fastchat.serve.test_message --model-name {model}

# Start the RESTful API service：
python -m fastchat.serve.openai_api_server  --host localhost --port 8000
```

# link
- https://github.com/lm-sys/FastChat




