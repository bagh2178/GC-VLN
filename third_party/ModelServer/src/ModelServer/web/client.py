import requests
import json
import pickle
import base64
from typing import List, Any, Dict, Optional
import time
import threading
import uuid


class ModelClient:
    """通用模型客户端类，用于与FastAPI服务器通信"""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        """
        初始化ModelClient
        
        Args:
            server_url: 服务器地址，默认为 http://localhost:8000
        """
        self.server_url = server_url.rstrip('/')
        self.async_mode = AsyncMode(self)  # 初始化异步客户端
    
    def _make_request(self, endpoint: str, data: Any = None) -> Dict[str, Any]:
        """
        发送HTTP请求到服务器
        
        Args:
            endpoint: API端点
            data: 请求数据
            
        Returns:
            服务器响应数据
        """
        url = f"{self.server_url}{endpoint}"
        
        try:
            if data:
                response = requests.post(url, data=data)
            else:
                response = requests.get(url)
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"请求失败: {str(e)}")
        except json.JSONDecodeError as e:
            raise Exception(f"响应解析失败: {str(e)}")
    
    def __getattr__(self, name):
        """
        动态捕获未定义的方法调用，序列化name/args/kwargs分别发送到服务端。
        """
        def wrapper(*args, **kwargs):
            try:
                base64_args = base64.b64encode(pickle.dumps(args)).decode('utf-8')
                base64_kwargs = base64.b64encode(pickle.dumps(kwargs)).decode('utf-8')
            except Exception as e:
                raise Exception(f"参数序列化失败: {str(e)}")
            data = {
                "name": name,
                "args": base64_args,
                "kwargs": base64_kwargs
            }
            result = self._make_request("/api/process", data)
            if result.get("success"):
                try:
                    base64_result = result.get("result")
                    if base64_result:
                        decoded_data = base64.b64decode(base64_result.encode('utf-8'))
                        deserialized_result = pickle.loads(decoded_data)
                        return deserialized_result
                    else:
                        return None
                except Exception as e:
                    raise Exception(f"结果反序列化失败: {str(e)}")
            else:
                raise Exception(f"命令执行失败: {result.get('error', '未知错误')}")
        return wrapper
    
    def __call__(self, *args, **kwargs):
        """
        使客户端对象可调用
        """
        return self.__getattr__('__call__')(*args, **kwargs) 


class Proxy:
    """异步任务代理对象，保存token、状态、结果等"""
    def __init__(self, token: str, name: str = None):
        self.token = token
        self.name = name
        self.created_at = time.time()
        self.finished_at = None
        self._done = threading.Event()
        self._result = None
        self._error = None
        self._returned = False  # 是否已返回结果
        self._result_wait_time = None  # 结果等待时间

    def set_result(self, result):
        self._result = result
        self.finished_at = time.time()
        self._result_wait_time = self.finished_at - self.created_at
        self._done.set()

    def set_error(self, error):
        self._error = error
        self.finished_at = time.time()
        self._result_wait_time = self.finished_at - self.created_at
        self._done.set()

    def is_done(self):
        return self._done.is_set()

    @property
    def result(self):
        """
        获取异步任务的结果，等待最多600秒。
        Returns:
            任务结果
        Raises:
            TimeoutError: 超时未完成
            Exception: 任务执行异常
        """
        return self._get_result_with_timeout(600)

    def _get_result_with_timeout(self, timeout: float = 600):
        finished = self._done.wait(timeout)
        if not finished:
            raise TimeoutError("任务等待超时（10分钟）")
        if self._error:
            result = self._error
            self._returned = True  # 标记已返回
            self._clear()
            raise result
        result = self._result
        self._returned = True  # 标记已返回
        self._clear()
        return result

    def _clear(self):
        self._result = None
        self._error = None

    def __str__(self):
        now = time.time()
        total_waited = now - self.created_at
        result_waited = self._result_wait_time if self._result_wait_time is not None else (self.finished_at - self.created_at if self.finished_at else None)
        info = [
            f"AsyncResult(token={self.token}",
            f"name={self.name}",
            f"created_at={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.created_at))}",
            f"done={self.is_done()}"
        ]
        if self.finished_at:
            info.append(f"finished_at={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.finished_at))}")
        info.append(f"total_waited={total_waited:.2f}s")
        info.append(f"result_waited={result_waited:.2f}s" if result_waited is not None else "result_waited=None")
        info.append(f"returned={self._returned}")
        info.append(")")
        return ", ".join(info)

    __repr__ = __str__

    def __call__(self, timeout: float = 600):
        result = self._get_result_with_timeout(timeout=timeout)
        self._returned = True  # 标记已返回
        return result


class AsyncMode:
    """异步模型客户端类，用于与FastAPI服务器通信"""
    def __init__(self, model_client: ModelClient):
        self.server_url = model_client.server_url
        self._model_client = model_client  # 直接引用传入的ModelClient对象

    def __getattr__(self, name):
        def wrapper(*args, **kwargs):
            token = str(uuid.uuid4())
            proxy = Proxy(token, name=name)
            def target():
                try:
                    func = getattr(self._model_client, name)
                    server_result = func(*args, **kwargs)
                    proxy.set_result(server_result)
                except Exception as e:
                    proxy.set_error(e)
            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            return proxy
        return wrapper
    
    def __call__(self, *args, **kwargs):
        """
        使异步模式对象可调用
        """
        return self.__getattr__('__call__')(*args, **kwargs)

