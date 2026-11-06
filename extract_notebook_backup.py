"""Extract the zip that a Colab cell printed between QCR_BACKUP markers into a local folder.

Usage: python extract_notebook_backup.py <notebook.ipynb> [out_dir]   (PubMedCLIP or BiomedCLIP notebook)
"""
import base64
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path, PurePosixPath

BEGIN, END = '===QCR_BACKUP_BEGIN===', '===QCR_BACKUP_END==='


def main():
    nb_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('colab_backup')
    nb = json.loads(nb_path.read_text(encoding='utf-8'))
    payload = None
    for cell in nb['cells']:
        for out in cell.get('outputs', []):
            text = out.get('text')
            if text is None:
                text = out.get('data', {}).get('text/plain', '')
            text = ''.join(text) if isinstance(text, list) else text
            if BEGIN in text and END in text:
                payload = text.split(BEGIN, 1)[1].split(END, 1)[0]  # keep the last one found
    if payload is None:
        sys.exit(f'No backup block found in {nb_path}. Save the notebook (Ctrl+S) after the cell finishes.')
    header, b64 = payload.strip().split('\n', 1)
    expected = header.split('sha256=', 1)[1].strip()
    data = base64.b64decode(''.join(b64.split()))
    if hashlib.sha256(data).hexdigest() != expected:
        sys.exit('SHA-256 mismatch: the output was truncated or edited.')
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for member in archive.infolist():
            name = PurePosixPath(member.filename)
            if name.is_absolute() or '..' in name.parts:
                sys.exit('Unsafe archive member: ' + member.filename)
        archive.extractall(out_dir)
        for member in archive.infolist():
            print(f'{member.file_size:>10,}  {member.filename}')
    print('extracted to', out_dir.resolve())


if __name__ == '__main__':
    main()
