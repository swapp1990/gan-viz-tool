import threading
import queue
active_queues = []
class Worker(threading.Thread):
    def __init__(self, id, modelCls=None, socketio=None):
        threading.Thread.__init__(self)
        self.mailbox = queue.Queue()
        self.id = id
        self.modelCls = modelCls
        self.socketio = socketio
        print("thread ", self.id)
        active_queues.append(self.mailbox)
    
    def run(self):
        while True:
            data = self.mailbox.get()
            if 'action' in data.keys():
                print(self, 'received a message', data['action'], str(data['id']))
                if self.id == data['id']:
                    self.doWork(data)
            elif 'log' in data.keys():
                print(self, 'received a message', data['log'], str(data['id']))
                if self.id == data['id']:
                    self.emitLogs(data)     

    def doWork(self, msg):
        print("do work ", self.id)
        if(self.id == 0):
            self.modelCls.doWork(msg)
        else:
            self.emitGeneral(msg)
    
    def emitLogs(self, msg):
        print("emit logs ", msg)
        self.socketio.emit('logs', msg)
    
    def emitGeneral(self, msg):
        #print('emitGeneral ', msg)
        self.socketio.emit('General', msg)

    def stop(self):
        self.mailbox.put("shutdown")
        self.join()

def broadcast_event(data):
    for q in active_queues:
        q.put(data)
