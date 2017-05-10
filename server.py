import json
import socket

from engine.engine import Engine

import pdb

class Server(object):

    def __init__(self, ip, port):
        self.bufsz = 36000
        self.ip = ip
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.engine = None

    def run(self):
        self.engine = Engine('/data/UCF/demo/Engine_Service/engine')
        self.socket.bind((self.ip, self.port))
        self.socket.listen(1)

        print('Engine server is now running on port %d' % self.port)

        while True:
            conn, addr = self.socket.accept()
            print('Accepted connection %s' % str(addr))

            while True:
                indexList = json.loads(conn.recv(self.bufsz))
                result = self.engine.processQuery(indexList).tolist()
                result = result[:300]
                conn.send(json.dumps(result))

if __name__ == '__main__':
    ip = '127.0.0.1'
    port = 8005

    server = Server(ip, port)
    server.run()
