# 🤖 ModelServer
## Connect Everything: Remote Function and Class Calling

[![中文](https://img.shields.io/badge/README-中文-red)](README.md)

> **ModelServer** is a web-based service used for deploying models on the host computer or isolating the environment between multiple models, and it is a framework that runs services called by clients through APIs.

## 📦 Installation

### Method 1: Install via pip
```bash
pip install git+https://github.com/bagh2178/ModelServer.git
```

### Method 2: Clone and Install
```bash
git clone https://github.com/bagh2178/ModelServer.git
pip install -e ModelServer/
```

## 🚀 Usage

### 🖥️ Server

#### Method 1: Quick Start
```bash
python scripts/quickstart_server/hexmove.py
```

#### Method 2: Programmatic Start
```python
from ModelServer import start_server

# Create model instance
your_model = YourModelClass(param1="value1", param2="value2")  # Pass parameters as needed

# Start server with model instance
start_server(your_model, port=7002)  # Pass initialized model instance and server port
```

### 💻 Client

#### Initialize Client

**Method 1: Quick Start**
```python
from ModelServer import hexmove
```

**Method 2: Programmatic Start**
```python
from ModelServer import ModelClient

# Local model
your_model = ModelClient('http://localhost:7002')

# Remote model
your_model = ModelClient('http://166.111.73.73:7002')
```

#### Sync Mode Usage Example

The same client object (such as `your_model` or `hexmove`) supports both sync and async modes. The following shows how to use the sync mode: just call the method directly to get the result.

You can use `your_model` just like a regular instance of `YourModelClass` and directly call its methods on the client side.

```python
# 📷 Get RGB-D images
rgb, depth, pose, timestamp = hexmove.get_rgbd_image('336L_up', format='JPEG', pose=True)
```

#### Async Mode Usage Example

In async mode, calling `your_model.async_mode.xxx()` immediately returns a `proxy`. Call it to get the final result:

```python
# Async call through async_mode
proxy = hexmove.async_mode.get_rgbd_image('336L_up', format='JPEG', pose=True)

# Other code can be executed here

# Wait and get result (blocking wait for result)
rgb, depth, pose, timestamp = proxy()  # or proxy.result
```

## 🔧 Features

- 🌐 **Web Service Architecture** - HTTP API-based model service
- 🔒 **Environment Isolation** - Support for multi-model environment isolation deployment
- 📡 **Remote Calling** - Support for local and remote model calling
- ⏩ **Async Support** - Support for asynchronous model inference and calling
- 🔌 **Easy Integration** - Simple client API interface

## 🤝 Contributing

Welcome to submit Issues and Pull Requests!

## 📄 License

This project is licensed under the MIT License. 