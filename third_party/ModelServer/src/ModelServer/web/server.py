import asyncio
import base64
import json
import pickle
import threading
import time
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# 创建FastAPI应用实例
app = FastAPI(
    title="Model Server",
    description="一个通用的FastAPI服务器，可以处理任何模型",
    version="0.3.0"
)

# 配置CORS中间件，允许所有来源
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有请求头
)

# 全局模型实例，由启动器设置
model_instance: Any = None
# 创建线程池用于异步处理模型调用
# 虽然Python有GIL限制，但对于I/O密集型任务（如硬件等待）仍然有效
# 硬件交互大部分时间在等待，此时会释放GIL，其他线程可以执行
thread_pool = ThreadPoolExecutor(max_workers=64, thread_name_prefix="model_executor")


def set_model_instance(instance: Any) -> None:
    """设置模型实例"""
    global model_instance
    model_instance = instance


@app.get("/")
async def root() -> Dict[str, str]:
    """根路径，显示API信息"""
    return {
        "message": "Model Server API",
        "docs": "/docs",
        "process": "/api/process"
    }


@app.post("/api/process")
async def process_request(
    name: str = Form(...),
    args: str = Form(...),
    kwargs: str = Form(...)
) -> Dict[str, Union[bool, str]]:
    """处理POST请求，使用模型实例处理数据"""
    try:
        # 解码args和kwargs
        try:
            decoded_args = base64.b64decode(args.encode('utf-8'))
            decoded_kwargs = base64.b64decode(kwargs.encode('utf-8'))
            args_list = pickle.loads(decoded_args)
            kwargs_dict = pickle.loads(decoded_kwargs)
        except Exception as e:
            return {
                "success": False,
                "error": f"参数格式错误，无法解析: {str(e)}"
            }
        # 检查属性是否存在
        if not hasattr(model_instance, name):
            return {
                "success": False,
                "error": f"模型实例没有属性或方法: {name}"
            }
        
        loop = asyncio.get_event_loop()
        def model_call_wrapper():
            attr = getattr(model_instance, name)
            if callable(attr):
                return attr(*args_list, **kwargs_dict)
            else:
                raise AttributeError(f"{name} 不是可调用的方法，禁止访问属性")
                
        result = await loop.run_in_executor(thread_pool, model_call_wrapper)
        # 序列化结果
        try:
            serialized_result = pickle.dumps(result)
            base64_result = base64.b64encode(serialized_result).decode('utf-8')
            return {
                "success": True,
                "result": base64_result,
                "message": "命令执行成功"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"结果序列化失败: {str(e)}",
                "message": "命令执行失败"
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "命令执行失败"
        }


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:
    """启动服务器"""
    if reload:
        # 当启用热重载时，使用导入字符串
        # 注意：热重载会导致全局变量丢失，不推荐在生产环境中使用
        uvicorn.run(
            "ModelServer.fastapi.server:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
    else:
        # 当不启用热重载时，使用应用程序实例
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )


def start_server(model_instance, host="0.0.0.0", port=8000, reload=False):
    """
    Create model server

    Args:
        model_instance: Initialized model instance
        host: Server host address
        port: Server port
        reload: Enable hot reload (default False, avoid global variable loss)
    """
    print("🚀 Starting model server...")

    try:
        # Validate model instance
        if model_instance is None:
            raise ValueError("Model instance cannot be None")

        print(f"📦 Model type: {type(model_instance).__name__}")

        # Set model instance
        set_model_instance(model_instance)
        print("🔗 Model instance bound to server")

        # Start server
        print("🌐 Starting web server...")
        print(f"📍 Server address: http://{host}:{port}")
        print("=" * 50)

        run_server(host=host, port=port, reload=reload)

    except Exception as e:
        print(f"❌ Failed to start: {e}")
        sys.exit(1)