import os

output_file = 'all_code_for_ai.txt'
# Maine isme Next.js (.ts, .tsx) aur config files (.json, .txt) add kar di hain
allowed_extensions = ['.py', '.ts', '.tsx', '.js', '.jsx', '.json', '.mjs', '.txt', '.md'] 

# .next aur node_modules ko ignore karna zaroori hai warna file bohot badi ho jayegi
ignore_folders = ['venv', '__pycache__', '.git', 'node_modules', '.next']

with open(output_file, 'w', encoding='utf-8') as outfile:
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if d not in ignore_folders] # Faltu folders ignore karo
        for file in files:
            if any(file.endswith(ext) for ext in allowed_extensions):
                # Lock files ignore kar rahe hain kyunki unki zaroorat nahi hoti logic samajhne mein
                if 'package-lock' in file or 'pnpm-lock' in file:
                    continue
                    
                filepath = os.path.join(root, file)
                outfile.write(f"\n{'='*50}\n")
                outfile.write(f"File: {filepath}\n")
                outfile.write(f"{'='*50}\n\n")
                try:
                    with open(filepath, 'r', encoding='utf-8') as infile:
                        outfile.write(infile.read())
                except Exception as e:
                    outfile.write(f"Error reading file: {e}\n")

print(f"Bhai, saara code '{output_file}' mein save ho gaya hai!")