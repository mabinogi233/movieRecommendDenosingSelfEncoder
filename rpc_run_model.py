import time

import grpc

from concurrent import futures

from data.rpc_service import msg_pb2,msg_pb2_grpc

import run_model

HOST = "127.0.0.1"
IP = "7777"

class service(msg_pb2_grpc.SerServiceServicer):
    def rec(self, request, context):
        phonenum = request.phonenumber
        k = request.num
        run_model.prec(phonenum,k)
        return msg_pb2.outMsg(msg="finish")

def serve():
    grpcService = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    msg_pb2_grpc.add_SerServiceServicer_to_server(service(),grpcService)
    grpcService.add_insecure_port(HOST+":"+IP)
    grpcService.start()

    try:
        while(True):
            time.sleep(60 * 60 * 24 * 30)
    except Exception:
        grpcService.stop(0)

if __name__ == '__main__':
    serve()
