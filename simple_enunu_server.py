#! /usr/bin/env python3
# coding: utf-8
# fmt: off
print('Starting enunu server...')
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime

sys.path.append(os.path.dirname(__file__))
import numpy as np
import simple_enunu
try:
    import zmq
except ModuleNotFoundError:
    python_exe = os.path.join('.', 'python-3.9.13-embed-amd64', 'python.exe')
    command = [python_exe, '-m', 'pip', 'install', 'pyzmq']
    print('command:', command)
    subprocess.run(command, check=True)
    import zmq
# fmt: on

def check():
    return{
        'name': 'SimpleENUNUServer',
        'version': '0.4.0',
        'author': 'roku10shi',
    }

def timing(engine: simple_enunu.SimpleEnunu):
    print('timing: start')
    simple_enunu.run_timing(engine=engine)
    
    for path in (engine.path_full_timing, engine.path_mono_timing):
        if not os.path.isfile(path):
            raise Exception(f'{datetime.now()} :`{os.path.basename(path)}` does not exist.')
    print('timing: end')
    return {
        'path_full_timing': engine.path_full_timing,
        'path_mono_timing': engine.path_mono_timing,
    }


def acoustic(style_shift: int ,engine: simple_enunu.SimpleEnunu):
    print('acoustic: start')
    if not set_features(engine):
        simple_enunu.run_timing(engine=engine,step='acoustic')
        simple_enunu.run_acoustic(engine=engine,style_shift=int(style_shift))
        simple_enunu.run_npy(engine=engine)
    print('acoustic: end')
    return {
        'path_f0': engine.path_f0,
        'path_spectrogram': engine.path_spectrogram,
        'path_aperiodicity': engine.path_aperiodicity,
        'path_mel': engine.path_mel,
        'path_vuv': engine.path_vuv,
    }

def synthe(out_wav_path: str,engine: simple_enunu.SimpleEnunu):
    print('synthe: start')
    set_features(engine)
    simple_enunu.run_synthesizer(out_wav_path=out_wav_path,engine=engine)
    print('synthe: end')
    return {
        'path_wav': out_wav_path,
    }

def set_features(engine: simple_enunu.SimpleEnunu):
    if os.path.exists(engine.path_editorf0) or os.path.exists(engine.path_mel) or os.path.exists(engine.path_vuv):
        print('set_features: update features')
        f0 = np.load(engine.path_editorf0)
        f0 = np.array(f0).reshape(-1, 1)
        lf0 = f0.copy()
        lf0[np.nonzero(f0)] = np.log(f0[np.nonzero(f0)])
        mel = np.load(engine.path_mel)
        vuv = np.load(engine.path_vuv)
        # 要素を代入
        multistream_features_list = [mel, lf0, vuv]
        # リストをタプルに戻す
        engine.multistream_features = tuple(multistream_features_list)
        return True
    elif os.path.exists(engine.path_f0) or os.path.exists(engine.path_mel) or os.path.exists(engine.path_vuv):
        print('set_features: update features')
        f0 = np.load(engine.path_f0)
        f0 = np.array(f0).reshape(-1, 1)
        lf0 = f0.copy()
        lf0[np.nonzero(f0)] = np.log(f0[np.nonzero(f0)])
        mel = np.load(engine.path_mel)
        vuv = np.load(engine.path_vuv)
        # 要素を代入
        multistream_features_list = [mel, lf0, vuv]
        # リストをタプルに戻す
        engine.multistream_features = tuple(multistream_features_list)
        return True
    return False

def poll_socket(socket, timetick = 100):
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    # wait up to 100msec
    try:
        while True:
            obj = dict(poller.poll(timetick))
            if socket in obj and obj[socket] == zmq.POLLIN:
                yield socket.recv()
    except KeyboardInterrupt:
        pass
    # Escape while loop if there's a keyboard interrupt.


def main():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind('tcp://*:15556')
    print('Started enunu server')


    #support = False
    engine_dict = {}

    for message in poll_socket(socket):
        """
        request body
        request[0]:step,
        request[1]:ust_path,
        request[2]:wav_path,
        request[3]:singer_name,
        request[4]:duration,
        request[5]:style_shift
        """
        request = json.loads(message)
        print('Received request: %s' % request)

        response = {}
        engine,duration,request_time = None,600,None
        try:
            if request[0] == 'ver_check':
                support = True
                response['result'] = check()
            elif support:
                if request[3] in engine_dict:
                    engine,duration,request_time = engine_dict[request[3]]
                    simple_enunu.updete_path(request[1],engine)
                else:
                    duration = int(request[4])
                    engine = simple_enunu.setup(request[1],engine,True)
                    request_time = time.time()
                    engine_dict[request[3]] = engine,duration,request_time


                if request[0] == 'timing':
                    response['result'] = timing(engine)
                elif request[0] == 'acoustic':
                    response['result'] = acoustic(request[5],engine)
                elif request[0] == 'synthe':
                    response['result'] = synthe(request[2],engine)
                else:
                    raise NotImplementedError('unexpected command %s' % request[1])
            
                keys_to_delete = [key for key in engine_dict.keys() if (time.time() - request_time ) > engine_dict[key][1]]
                for key in keys_to_delete:
                    del engine_dict[key]
            else:
                response['error'] = 'run ver_check.'
            
        except Exception as e:
            response['error'] = str(e)
            traceback.print_exc()

        print('Sending response: %s' % response)
        socket.send_string(json.dumps(response))


if __name__ == '__main__':
    main()
