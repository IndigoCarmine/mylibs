import os
import subprocess

folder = os.path.join(os.path.dirname(__file__), 'documents_for_llm')

for filename in os.listdir(folder):
    if filename.endswith('.py'):
        filepath = os.path.join(folder, filename)
        print(f'Running {filepath}...')
        subprocess.run(['python', filepath])