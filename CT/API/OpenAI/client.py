import socket
import threading
from openai import OpenAI

# socket(client)
class ModelClient:
    def __init__(self, host='localhost', port=8124):
        self.host = host
        self.port = port

    def send_request(self, input_text):
        try:
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.connect((self.host, self.port))

            encoded_input = input_text.encode('utf-8')
            client.send(len(encoded_input).to_bytes(4, 'big') + encoded_input)

            raw_length = client.recv(4)
            if len(raw_length) != 4:
                return "Invalid response"
            response_length = int.from_bytes(raw_length, 'big')

            data = b''
            while len(data) < response_length:
                packet = client.recv(response_length - len(data))
                if not packet:
                    break
                data += packet

            return data.decode('utf-8')
        except Exception as e:
            return f"Error: {str(e)}"
        finally:
            client.close()

# OpenAI(server)
class OpenAIServer:
    def __init__(self, host='0.0.0.0', port=8123):
        self.host = host
        self.port = port
        self.fpga_client = ModelClient()  
        self.openai_client = OpenAI(api_key='xxx')  

    def handle_client(self, client_socket):
        try:
            # 带长度前缀）
            raw_length = client_socket.recv(4)
            if len(raw_length) != 4:
                return
            length = int.from_bytes(raw_length, 'big')
            data = b''
            while len(data) < length:
                packet = client_socket.recv(length - len(data))
                if not packet:
                    break
                data += packet
            user_input = data.decode('utf-8')

            
            ai_response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "将用户指令转换为FPGA命令。例如：'打开红灯' -> 'SET GPIO1 HIGH'"},
                    {"role": "user", "content": user_input}
                ]
            )
            command = ai_response.choices[0].message.content.strip()

            fpga_response = self.fpga_client.send_request(command)

            # 返回带长度前缀
            encoded_response = fpga_response.encode('utf-8')
            client_socket.send(len(encoded_response).to_bytes(4, 'big') + encoded_response)

        except Exception as e:
            error_msg = f"Server Error: {str(e)}".encode('utf-8')
            client_socket.send(len(error_msg).to_bytes(4, 'big') + error_msg)
        finally:
            client_socket.close()

    def start(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((self.host, self.port))
        server.listen(5)
        print(f"[*] OpenAI中间服务器已启动，监听 {self.host}:{self.port}")

        while True:
            client_sock, addr = server.accept()
            print(f"[+] 接受来自 {addr[0]}:{addr[1]} 的连接")
            client_handler = threading.Thread(
                target=self.handle_client,
                args=(client_sock,)
            )
            client_handler.start()

if __name__ == "__main__":
    server = OpenAIServer()
    server.start()