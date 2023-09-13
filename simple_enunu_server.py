#! /usr/bin/env python3
# coding: utf-8
# fmt: off
print('Starting enunu server...')
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime

sys.path.append(os.path.dirname(__file__))
import numpy as np
import enunu_steps
import enulib
try:
    import zmq
except ModuleNotFoundError:
    python_exe = os.path.join('.', 'python-3.9.13-embed-amd64', 'python.exe')
    command = [python_exe, '-m', 'pip', 'install', 'pyzmq']
    print('command:', command)
    subprocess.run(command, check=True)
    import zmq
# fmt: on


def timing(path_ust: str):
    config, temp_dir,engine = enunu_steps.setup(path_ust)
    path_full_timing, path_mono_timing = enunu_steps.run_timing(config, temp_dir,engine)
    
    for path in (path_full_timing, path_mono_timing):
        if not os.path.isfile(path):
            raise Exception(f'{datetime.now()} :`{os.path.basename(path)}` does not exist.')

    return {
        'path_full_timing': path_full_timing,
        'path_mono_timing': path_mono_timing,
    }


def acoustic(path_ust: str):
    config, temp_dir,engine = enunu_steps.setup(path_ust)
    
    path_full_timing, path_mono_timing = enunu_steps.run_timing(config, temp_dir,engine)

    path_acoustic, path_f0, path_spectrogram, \
        path_aperiodicity = enunu_steps.run_acoustic(config, temp_dir,engine)
    
    

    for path in (path_f0, path_spectrogram, path_aperiodicity):
        if os.path.isfile(path):
            arr = np.loadtxt(path, delimiter=',', dtype=np.float64)
            np.save(path[:-4] + '.npy', arr)
            os.remove(path)
    return {
        'path_acoustic': path_acoustic,
        'path_f0': path_f0,
        'path_spectrogram': path_spectrogram,
        'path_aperiodicity': path_aperiodicity,
    }

def vocoder(path_ust: str,out_wav_path: str):
    config, temp_dir,engine = enunu_steps.setup(path_ust)
    
    path_full_timing, path_mono_timing = enunu_steps.run_timing(config, temp_dir,engine)

    path_wav = enunu_steps.run_synthesizer(config, temp_dir,out_wav_path,engine)

    
    
    return {
        'path_wav': path_wav,
    }


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
    socket.bind('tcp://*:15555')
    print('Started enunu server')

    for message in poll_socket(socket):
        request = json.loads(message)
        print('Received request: %s' % request)

        response = {}
        try:
            if request[0] == 'timing':
                response['result'] = timing(request[1])
            elif request[0] == 'acoustic':
                response['result'] = acoustic(request[1])
            elif request[0] == 'vocoder':
                response['result'] = vocoder(request[1],request[2])
            else:
                raise NotImplementedError('unexpected command %s' % request[0])
        except Exception as e:
            response['error'] = str(e)
            traceback.print_exc()

        print('Sending response: %s' % response)
        socket.send_string(json.dumps(response))


if __name__ == '__main__':
    main()
