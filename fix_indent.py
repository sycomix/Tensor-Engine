
import sys
import re

path = 'examples/chat_llama.py'
content = open(path).read()

# Locate the fallback loop inside the layer loop
# We want to match the block starting from '# Check for shape mismatch' down to 'assigned += 1'
# and fix its indentation.

old_block_pattern = r'(\s+)# Check for shape mismatch.*?assigned \+= 1'
# We'll do a simpler replacement of the whole block from 'except Exception' to 'assigned += 1'

block_start = '                        except Exception:'
block_end = '                        assigned += 1'

# Find the block carefully
start_idx = content.find(block_start)
if start_idx == -1:
    print("Failed to find block start")
    sys.exit(1)

# Find the next 'assigned += 1' after start_idx
end_idx = content.find(block_end, start_idx)
if end_idx == -1:
    print("Failed to find block end")
    sys.exit(1)

end_idx += len(block_end)

new_block = '''                        except Exception:
                            pass

                        # Convert to numpy if needed
                        if isinstance(src_data, list):
                            try:
                                # Try to guess shape from len
                                if len(src_data) == param.shape[0] * param.shape[1]:
                                    # Assume src is [Out, In] -> [param.shape[1], param.shape[0]] 
                                    arr = np.array(src_data, dtype=np.float32).reshape(param.shape[1], param.shape[0])
                                else:
                                     # Fallback
                                     arr = np.array(src_data, dtype=np.float32).reshape(src_tensor.shape)
                            except Exception:
                                arr = np.array(src_data, dtype=np.float32)
                        else:
                            arr = src_data

                        # ALWAYS Transpose 2D weights for TensorEngine (Linear expects [In, Out])
                        # SafeTensors stores [Out, In]
                        if len(arr.shape) == 2:
                            # Start with assuming we need to transpose
                            # For Non-Square, we MUST transpose if src is [Out, In].
                            logger.debug("Transposing tensor %s: %s -> T", name, arr.shape)
                            arr = arr.T
                            src_data = arr.ravel().tolist()
                        elif isinstance(arr, np.ndarray):
                            src_data = arr.ravel().tolist()

                        param.set_data(src_data)
                        assigned += 1'''

new_content = content[:start_idx] + new_block + content[end_idx:]

with open(path, 'w') as f:
    f.write(new_content)

print("SUCCESSFULLY FIXED INDENTATION")
