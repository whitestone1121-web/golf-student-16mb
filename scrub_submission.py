import zipfile, re, sys
z = zipfile.ZipFile('golf/golf_submission.zip')
sensitive = re.compile(
    r'(/tmp/apex|alans/neural|paliquant|sk-[a-zA-Z0-9]{20}|api[_.]?key\s*=\s*["\'][^"\']+|trading_api|stress_mem\.db|patent\s*#)',
    re.I
)
hits = []
for name in z.namelist():
    if name.endswith(('.py', '.md', '.json')):
        txt = z.read(name).decode('utf-8', errors='ignore')
        lines = [(i+1, l.strip()) for i, l in enumerate(txt.splitlines()) if sensitive.search(l)]
        if lines:
            hits.append((name, lines))

if hits:
    for fname, ls in hits:
        print(f'\n⚠ LEAK in {fname}:')
        for lineno, l in ls:
            print(f'  L{lineno}: {l}')
    sys.exit(1)
else:
    print('✅ CLEAN — no sensitive strings found in golf_submission.zip')
