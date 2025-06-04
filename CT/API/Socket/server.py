import socket
import threading
from queue import Queue
import struct
import time

# 配置参数
OTHER_SERVER_HOST = "10.20.108.87"
OTHER_SERVER_PORT = 8124
other_server_socket = None
message_queue = Queue()
END_MARKER = b"<|END|>"  # 消息结束标记


def other_server_communication():
    """监听H100服务器连接并接收消息"""
    global other_server_socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(("0.0.0.0", OTHER_SERVER_PORT))
    server_socket.listen(1)
    print(f"Listening for H100 connections on port {OTHER_SERVER_PORT}...")

    while True:
        try:
            client_sock, addr = server_socket.accept()
            print(f"Accepted connection from H100: {addr}")
            other_server_socket = client_sock

            while True:
                # 读取消息长度
                length_data = other_server_socket.recv(4)
                if not length_data:
                    raise ConnectionError("Connection closed")

                # 解析消息长度
                msg_length = struct.unpack('>I', length_data)[0]

                # 读取完整消息
                chunks = []
                bytes_received = 0
                while bytes_received < msg_length:
                    chunk = other_server_socket.recv(min(msg_length - bytes_received, 4096))
                    if not chunk:
                        raise ConnectionError("Connection interrupted")
                    chunks.append(chunk)
                    bytes_received += len(chunk)

                # 处理消息内容
                full_message = b''.join(chunks).decode('utf-8')
                if full_message.endswith(END_MARKER.decode('utf-8')):
                    full_message = full_message[:-len(END_MARKER)]

                print(f"Received message (length: {msg_length}): {full_message[:50]}...")
                message_queue.put(full_message)

        except (ConnectionError, socket.error) as e:
            print(f"Server connection error: {str(e)}")
            if other_server_socket:
                other_server_socket.close()
                other_server_socket = None
            time.sleep(5)

        except Exception as e:
            print(f"Error processing message: {str(e)}")
            if other_server_socket:
                other_server_socket.close()
                other_server_socket = None
            time.sleep(1)


def send_to_other_server(message):
    """发送响应到H100服务器"""
    global other_server_socket
    if other_server_socket:
        try:
            # 添加结束标记
            full_message = message + END_MARKER.decode('utf-8')
            encoded_msg = full_message.encode('utf-8')

            # 发送消息长度
            msg_length = len(encoded_msg)
            other_server_socket.sendall(struct.pack('>I', msg_length))

            # 发送消息内容
            other_server_socket.sendall(encoded_msg)
            print(f"Sent response to server (length: {msg_length}): {message[:50]}...")

        except Exception as e:
            print(f"Failed to send response: {str(e)}")
            other_server_socket.close()
            other_server_socket = None


def main_response_handler():
    """主消息处理循环"""
    # 启动服务器监听线程
    server_thread = threading.Thread(target=other_server_communication, daemon=True)
    server_thread.start()

    while True:
        # 检查是否有新消息
        if not message_queue.empty():
            query = message_queue.get()
            print(f"Processing message: {query[:50]}...")

            # 固定响应内容
            response = "我是chatglm大模型。"
            print(f"Sending response: {response}")

            # 发送响应
            send_to_other_server(response)

        time.sleep(0.1)  # 避免CPU空转


if __name__ == "__main__":
    main_response_handler()