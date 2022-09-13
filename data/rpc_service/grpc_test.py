import grpc

from data.rpc_service import msg_pb2,msg_pb2_grpc

HOST = "127.0.0.1"
IP = "7777"

def run():
    conn = grpc.insecure_channel(HOST+":"+IP)
    client = msg_pb2_grpc.SerServiceStub(channel=conn)
    response = client.rec(msg_pb2.inMsg(phonenumber="19933055675",num=8))
    print(response.msg)

if __name__ == '__main__':
    run()
