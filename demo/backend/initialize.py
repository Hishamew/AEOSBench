__all__ = []
import os
import subprocess
from uuid import uuid4

from flask import jsonify, request

from constellation.data import Constellation, TaskSet

from .app import CACHE_ROOT, app


@app.route('/initialize_simulation', methods=['POST'])
def initialize_simulation():
    data = request.get_json()

    try:
        constellation = Constellation.from_dict(data['constellation'])
        tasksets = TaskSet.from_dicts(data['tasksets'])
        greedy = data.get('greedy', False)
    except TypeError as e:
        return jsonify({
            'error': str(e),
            'message': 'Invalid constellation or tasksets'
        }), 400

    uid = str(uuid4())
    try:
        os.makedirs(CACHE_ROOT / uid)
        constellation.dump(CACHE_ROOT / uid / 'constellation.json')
        tasksets.dump(CACHE_ROOT / uid / 'taskset.json')
    except OSError as e:
        return jsonify({
            'error': str(e),
            'message': 'Failed to create cache directory'
        }), 500

    run_simulation(uid, greedy)
    return jsonify({'message': 'Simulation started'}), 200


def run_simulation(uuid: str, greedy: bool) -> None:
    command = (
        f"python -m demo.run_simulation --thread_id '{uuid}'"
        f" {'--greedy' if greedy else ''} > {str(CACHE_ROOT/ uuid/'log.txt')} 2>&1"
    )

    subprocess.Popen(
        command,
        shell=True,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
