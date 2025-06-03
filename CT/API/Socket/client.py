import socket


class ModelClient:
    def __init__(self, host='localhost', port=8124):
        self.host = host
        self.port = port

    def send_request(self, input_text):
        try:
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.connect((self.host, self.port))

            # 发送请求
            encoded_input = input_text.encode('utf-8')
            client.send(len(encoded_input).to_bytes(4, 'big') + encoded_input)

            # 接收响应
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


if __name__ == "__main__":
    client = ModelClient()
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ('exit', 'quit'):
                break
            response = client.send_request(user_input)
            print("Assistant:", response)
        except KeyboardInterrupt:
            break