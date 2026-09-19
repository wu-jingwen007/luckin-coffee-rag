import subprocess
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent


def read_env_value(name, default=''):
    env_file = PROJECT_ROOT / '.env'
    if not env_file.exists():
        return default
    with open(env_file, 'r', encoding='utf-8-sig') as f:
        for line in f:
            if line.startswith(f'{name}='):
                return line.split('=', 1)[1].strip().strip('"').strip("'")
    return default


def run_preprocess():
    print('\n' + '=' * 60)
    print('Step 1/2: Running data preprocessing')
    print('=' * 60)
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / 'rag_app' / 'preprocess.py')],
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode != 0:
        print('\nPreprocessing failed. Please check errors above.')
        sys.exit(1)

def run_api_server():
    api_port = read_env_value('API_PORT', '8000')
    print('\n' + '=' * 60)
    print('Step 2/2: Starting API server')
    print('=' * 60)
    print(f'\nAPI URL: http://localhost:{api_port}')
    print(f'API Docs: http://localhost:{api_port}/docs')
    print('Frontend: http://localhost:8001')
    print('\nPress Ctrl+C to stop\n')
    
    api_proc = subprocess.Popen(
        [sys.executable, '-m', 'uvicorn', 'rag_app.api.app:app',
         '--host', '0.0.0.0', '--port', api_port, '--reload'],
        cwd=str(PROJECT_ROOT),
    )
    
    import threading
    import time
    import http.server
    
    def serve_frontend():
        time.sleep(2)
        os.chdir(str(PROJECT_ROOT / 'frontend'))
        handler = http.server.SimpleHTTPRequestHandler
        server = http.server.HTTPServer(('0.0.0.0', 8001), handler)
        print('Frontend server started: http://localhost:8001/index.html')
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.shutdown()
    
    t = threading.Thread(target=serve_frontend, daemon=True)
    t.start()
    
    try:
        api_proc.wait()
    except KeyboardInterrupt:
        print('\nStopping services...')
        api_proc.terminate()

if __name__ == '__main__':
    env_file = PROJECT_ROOT / '.env'
    if not env_file.exists():
        print('Error: .env file not found')
        sys.exit(1)
    
    api_key = read_env_value('DEEPSEEK_API_KEY')
    
    if not api_key or api_key == 'your_deepseek_api_key_here':
        print('Warning: DEEPSEEK_API_KEY not configured')
        sys.exit(1)
    
    database_file = PROJECT_ROOT / 'chroma_db' / 'chroma.sqlite3'
    if database_file.exists():
        print(f'Existing vector database found: {database_file}')
        print('Skipping preprocessing. Run rag_app/preprocess.py after changing documents.')
    else:
        run_preprocess()
    run_api_server()
