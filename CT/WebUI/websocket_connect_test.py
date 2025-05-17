import websocket
import base64
import json

def on_message(ws, message):
    print("收到消息:", message)

def on_error(ws, error):
    print("错误:", error)

def on_close(ws, close_status_code, close_msg):
    print("连接关闭")

def on_open(ws):
    print("连接成功")
    # 发送一个测试消息
    ws.send(json.dumps({"action": "ping"}))

if __name__ == "__main__":
    token = base64.b64encode(b"admin:yuhaolab.CT").decode()
    url = f"ws://10.20.108.87:7678/ws?token={token}"
    ws = websocket.WebSocketApp(url,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.run_forever()
