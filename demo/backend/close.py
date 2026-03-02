__all__ = []
from shutil import rmtree

from flask import jsonify, request

from .app import CACHE_ROOT, app


@app.route('/close_simulation', methods=['POST'])
def close():
    data = request.get_json()

    thread_id: str = data['thread_id']
    thread_root = CACHE_ROOT / thread_id

    if not thread_root.exists():
        return jsonify({
            'error': 'Invalid thread id or simulation is already closed'
        }), 400

    rmtree(thread_root)
    return jsonify({'message': 'Simulation closed successfully'}), 200
