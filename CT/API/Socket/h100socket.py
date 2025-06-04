import socket
import struct
import threading
import time
import queue
from flask import Flask, request, jsonify
import json

TCP_HOST = "10.20.108.87"   # 上位机IP地址
TCP_PORT = 8124            # 上位机端口
HTTP_PORT = 7000           
END_MARKER = b"<|END|>"    
BUFFER_SIZE = 4096         # 网络缓冲区大小
openai_model = "chatglm"
RECONNECT_DELAY = 5        # 重连延迟(秒)

fpga_connection = None     
request_queue = queue.Queue()
app = Flask(__name__)
connection_lock = threading.Lock()  
class FPGAConnection:
    def __init__(self):
        self.conn = None
        self.shutdown_event = threading.Event()
        self.lock = threading.Lock()
        
    def connect(self):
        
        while not self.shutdown_event.is_set():
            try:
                print(f"[H100] 尝试连接上位机 {TCP_HOST}:{TCP_PORT}")
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(10)
                sock.connect((TCP_HOST, TCP_PORT))
                self.conn = sock
                print(f"[H100] 成功连接到上位机")
                return True
            except Exception as e:
                print(f"[H100] 连接失败: {str(e)}")
                time.sleep(RECONNECT_DELAY)
        return False
    
    def send_message(self, message):
        
        if self.conn is None:
            raise ConnectionError("未连接到上位机")
            
        try:
            with self.lock:
                
                full_message = message + END_MARKER.decode('utf-8')
                encoded_msg = full_message.encode('utf-8')
                
                
                msg_length = len(encoded_msg)
                self.conn.sendall(struct.pack('>I', msg_length))
                
                
                self.conn.sendall(encoded_msg)
                print(f"[H100] 发送消息到上位机 (长度: {msg_length})")
                
                time.sleep(2)

                
                length_data = self.conn.recv(4)
                if not length_data:
                    raise ConnectionError("连接已关闭")
                msg_length = struct.unpack('>I', length_data)[0]
                
                
                chunks = []
                bytes_received = 0
                while bytes_received < msg_length:
                    chunk = self.conn.recv(min(msg_length - bytes_received, BUFFER_SIZE))
                    if not chunk:
                        raise ConnectionError("连接中断")
                    chunks.append(chunk)
                    bytes_received += len(chunk)
                    
                full_message = b''.join(chunks).decode('utf-8')
                
                
                if full_message.endswith(END_MARKER.decode('utf-8')):
                    full_message = full_message[:-len(END_MARKER)]
                    
                print(f"[H100] 收到上位机响应 (长度: {len(full_message)})")
                return full_message
                
        except Exception as e:
            print(f"[H100] 通信错误: {str(e)}")
            self.close()
            raise

    def close(self):
        """关闭连接"""
        if self.conn:
            try:
                self.conn.close()
            except:
                pass
            self.conn = None
        print("[H100] 连接已关闭")

def tcp_client():
    
    global fpga_connection
    
    while True:
        with connection_lock:
            if fpga_connection is None or fpga_connection.conn is None:
                fpga_connection = FPGAConnection()
                if not fpga_connection.connect():
                    print("[H100] 无法建立连接，稍后重试...")
                    time.sleep(RECONNECT_DELAY)
                    continue
        
        
        time.sleep(1)

def process_requests():
    global fpga_connection  
   
    while True:
        try:
            queue_item = request_queue.get()

            if len(queue_item) == 2:
                _, data = queue_item
                response_callback = None  
            elif len(queue_item) == 3:
                _, data, response_callback = queue_item
            else:
                print(f"[H100] 错误：无效的队列项格式: {queue_item}")
                continue
            
            
            with connection_lock:
                if fpga_connection is None or fpga_connection.conn is None:
                    print("[H100] 错误：未连接上位机")
                    continue
                    
            
            try:
                with connection_lock:
                    current_conn = fpga_connection

                send_data = {
                    "model": data.get("model"),
                    "messages": data.get("messages"),
                    "temperature": data.get("temperature"),
                    "max_tokens": data.get("max_tokens")
                }
                
                response = current_conn.send_message(json.dumps(send_data))
                
                
                openai_response = {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": openai_model,
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": response
                        },
                        "finish_reason": "stop",
                        "index": 0
                    }]
                }
                
                
                if response_callback:
                    response_callback(openai_response)
                    
            except Exception as e:
                print(f"[H100] 处理请求失败: {str(e)}")
                
                with connection_lock:
                    if fpga_connection:
                        fpga_connection.close()
                        fpga_connection = None
                
        except Exception as e:
            print(f"[H100] 请求处理错误: {str(e)}")
            time.sleep(1)

@app.route('/v1/chat/completions', methods=['POST'])
def openai_endpoint():
    
    try:
        
        with connection_lock:
            if fpga_connection is None or fpga_connection.conn is None:
                return jsonify({"error": "未连接上位机"}), 503
        
        
        request_data = request.json
        print(f"[H100] 收到Agent请求: {json.dumps(request_data, indent=2)}")

        response_event = threading.Event()
        response_data = [None]
        
        def response_callback(resp):
            response_data[0] = resp
            response_event.set()

        processing_data = {
            "messages": request_data['messages'],
            "model": request_data.get('model', 'glm'),
            "max_tokens": request_data.get('max_tokens', 1024),
            "temperature": request_data.get('temperature', 0.7),
            "response_callback": response_callback
        }

        request_queue.put((1, request_data, response_callback))
        print("发送给agent回复")
        #request_queue.put(("fpga", processing_data))
        response_event.wait(timeout=60)
        
        if response_data[0] is None:
            return jsonify({"error": "处理超时"}), 504
        
        return jsonify(response_data[0])
    
    except Exception as e:
        print(f"[H100] API错误: {str(e)}")
        return jsonify({"error": "服务器错误"}), 500

if __name__ == "__main__":
    
    tcp_thread = threading.Thread(target=tcp_client, daemon=True)
    tcp_thread.start()

    
    processing_thread = threading.Thread(target=process_requests, daemon=True)
    processing_thread.start()

    print(f"[H100] HTTP服务器启动，监听端口 {HTTP_PORT}")
    app.run(host="0.0.0.0", port=HTTP_PORT, threaded=True)