# Just checking if rocket has serde json support
# Based on Cargo.toml rocket version
import re
with open('Cargo.toml') as f:
    content = f.read()
m = re.search(r'rocket.*=.*"(\d+\.\d+\.\d+)"', content)
if m:
    print(f"Rocket version: {m.group(1)}")
