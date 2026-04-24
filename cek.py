python -c "
import zipfile, sys
with zipfile.ZipFile(r"C:\Users\ASUS\Documents\LocObj\project-5-at-2026-04-23-21-48-f1a4314d.zip", 'r') as z:
    files = z.namelist()
    print(f'Total file: {len(files)}')
    print('--- 40 file pertama ---')
    for f in files[:40]:
        print(f)
"