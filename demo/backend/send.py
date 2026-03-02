__all__ = []
from flask import jsonify, request
from todd.patches.py_ import json_load

from ..interface import Socket
from .app import CACHE_ROOT, app


@app.route("/get_result", methods=['POST'])
def get_result():
    data = request.get_json()

    thread_id: str = data['thread_id']
    thread_cache_root = CACHE_ROOT / thread_id

    if not (thread_cache_root).exists():
        return jsonify({'error': 'Invalid thread id '}), 400

    if (thread_cache_root / 'failed.txt').exists():
        return jsonify({'error': 'Simulation failed'}), 500

    time_step: int = data['time_step']
    target_file_path = thread_cache_root / 'result' / f'{time_step}.json'
    if not target_file_path.exists():
        return jsonify({'error': 'Simulation not complete yet'}), 500

    socket: Socket = json_load(target_file_path)

    trace = []

    for i in range(time_step):
        target_file_path = thread_cache_root / 'result' / f'{i}.json'
        record: Socket = json_load(str(target_file_path))
        satellites = record['satellites_data']

    return json_load(target_file_path), 200
